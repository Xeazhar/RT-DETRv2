"""
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import time 
import json
import datetime
import torch 

from ..misc import dist_utils, profiler_utils
from ._solver import BaseSolver
from .det_engine import train_one_epoch, evaluate

class DetSolver(BaseSolver):
    
    def fit(self, ):
        print("Start training")
        self.train()
        args = self.cfg

        n_parameters = sum([p.numel() for p in self.model.parameters() if p.requires_grad])
        print(f'number of trainable parameters: {n_parameters}')

        best_stat = {'epoch': -1, }
        
        # --- EARLY STOPPING CONFIG ---
        patience_limit = 20
        patience_counter = 0
        best_map_score = -1.0
        # -----------------------------

        start_time = time.time()
        start_epoch = self.last_epoch + 1

        for epoch in range(start_epoch, args.epoches):
            if hasattr(self.train_dataloader, 'set_epoch'):
                self.train_dataloader.set_epoch(epoch)
            
            if dist_utils.is_dist_available_and_initialized():
                if hasattr(self.train_dataloader.sampler, 'set_epoch'):
                    self.train_dataloader.sampler.set_epoch(epoch)
            
            train_stats = train_one_epoch(
                self.model, self.criterion, self.train_dataloader, 
                self.optimizer, self.device, epoch, 
                max_norm=args.clip_max_norm, print_freq=args.print_freq, 
                ema=self.ema, scaler=self.scaler, 
                lr_warmup_scheduler=self.lr_warmup_scheduler, writer=self.writer
            )

            if self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished():
                self.lr_scheduler.step()
            
            self.last_epoch += 1

            # Checkpoints (Last and Periodic)
            if self.output_dir:
                checkpoint_paths = [self.output_dir / 'last.pth']
                if (epoch + 1) % args.checkpoint_freq == 0:
                    checkpoint_paths.append(self.output_dir / f'checkpoint{epoch + 1:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    state_dict = self.state_dict()
                    self._strip_state_dict(state_dict)
                    dist_utils.save_on_master(state_dict, checkpoint_path)

            # Evaluation
            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module, self.criterion, self.postprocessor, 
                self.val_dataloader, self.evaluator, self.device
            )

            # --- EARLY STOPPING LOGIC ---
            # test_stats['coco_eval_bbox'][0] is the primary mAP metric
            current_map = test_stats['coco_eval_bbox'][0] if 'coco_eval_bbox' in test_stats else 0
            
            if current_map > best_map_score:
                best_map_score = current_map
                patience_counter = 0
                print(f"✨ Improvement! New best mAP: {best_map_score:.4f}")
            else:
                patience_counter += 1
                print(f"⏳ No improvement for {patience_counter}/{patience_limit} epochs.")

            # Best Model Logic
            for k in test_stats:
                if self.writer and dist_utils.is_main_process():
                    for i, v in enumerate(test_stats[k]):
                        self.writer.add_scalar(f'Test/{k}_{i}'.format(k), v, epoch)
            
                if k in best_stat:
                    if test_stats[k][0] > best_stat[k]:
                        best_stat['epoch'] = epoch
                        best_stat[k] = test_stats[k][0]
                else:
                    best_stat['epoch'] = epoch
                    best_stat[k] = test_stats[k][0]

                if best_stat['epoch'] == epoch and self.output_dir:
                    state_dict = self.state_dict()
                    self._strip_state_dict(state_dict)
                    dist_utils.save_on_master(state_dict, self.output_dir / 'best.pth')

            print(f'best_stat: {best_stat}')

            # Logging to text file
            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'test_{k}': v for k, v in test_stats.items()},
                'epoch': epoch,
                'n_parameters': n_parameters
            }

            if self.output_dir and dist_utils.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

            # --- THE FINAL STOP ---
            if patience_counter >= patience_limit:
                print(f"🛑 Early stopping triggered at epoch {epoch}. No improvement for {patience_limit} epochs.")
                break # This exits the epoch loop safely

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))
    def val(self, ):
        self.eval()
        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(module, self.criterion, self.postprocessor,
                self.val_dataloader, self.evaluator, self.device)
                
        if self.output_dir:
            dist_utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")
        
        return

    def _strip_state_dict(self, state_dict):
        # Use .get() to safely handle missing YAML keys
        save_optimizer = self.cfg.yaml_cfg.get('save_optimizer', False)
        save_ema = self.cfg.yaml_cfg.get('save_ema', True)

        if not save_optimizer and "optimizer" in state_dict:
            state_dict.pop("optimizer")
        
        if not save_ema and "ema" in state_dict:
            state_dict.pop("ema")