"""
Base Checkpoint Manager
======================

Domain-agnostic checkpoint management functionality.
Common to all GAN architectures (WaveGAN, DCGAN, etc.)

Implements optimized structure from metrics-checkpoints-how-to.txt:
- Rolling buffer strategy (max 2 checkpointy)
- Checkpoint co 10 epok
- Auto-cleanup najstarszych
- Best model tracking (najniższy avg_generator_loss)
- Final model backup
- Disaster recovery capabilities
"""

import os
import json
import shutil
import tarfile
import torch
import random
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from abc import ABC, abstractmethod


class CheckpointManagerBase(ABC):
    """
    Base class for checkpoint management across different GAN architectures
    
    Optimized structure according to metrics-checkpoints-how-to.txt:
    output_analysis/
      checkpoints/
        checkpoint_epoch_10.tar  # najnowszy (rolling buffer)
        checkpoint_epoch_20.tar  # poprzedni (max 2 jednocześnie)  
        best_model.tar (najlepszy avg_generator_loss - nie liczy się do bufora)
        final_model.tar (końcowy stan - nie liczy się do bufora)
    """
    
    def __init__(self, checkpoint_dir: str = "output_analysis/checkpoints", model_type: str = "base"):
        self.checkpoint_dir = checkpoint_dir
        self.model_type = model_type
        self.setup_directories()
        
        # Optimized checkpoint strategy
        self.checkpoint_frequency = 10  # Co 10 epok
        self.rolling_buffer_size = 2  # Max 2 checkpointy jednocześnie
        self.best_generator_loss = float('inf')
        self.checkpoint_history = []  # Track checkpoint files for rolling buffer
        
        print(f"💾 {model_type.upper()} Checkpoint manager initialized - directory: {checkpoint_dir}")
    
    def setup_directories(self):
        """Create optimized checkpoint directory structure"""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        print(f"📁 Checkpoint directory created: {self.checkpoint_dir}")
    
    def save_checkpoint(self, generator, discriminator,
                       optimizer_g, optimizer_d,
                       epoch: int, iteration: int,
                       generator_loss: float, discriminator_loss: float,
                       **domain_specific_data) -> str:
        """
        Save checkpoint with optimized rolling buffer strategy
        
        STRATEGIA MINIMALNA (ROLLING BUFFER):
        - checkpoint: co 10 epok (buffer 2 plików - rolling)
        - zawsze latest checkpoint + previous checkpoint
        - auto-cleanup: zapisanie nowego usuwa najstarszy (jeśli >2)
        - best_model: najniższy avg_generator_loss (osobno przechowywany)
        - final_model: ostatni stan treningu
        
        Args:
            generator: Generator model
            discriminator: Discriminator model  
            optimizer_g: Generator optimizer
            optimizer_d: Discriminator optimizer
            epoch: Current epoch
            iteration: Current iteration
            generator_loss: Current generator loss
            discriminator_loss: Current discriminator loss
            **domain_specific_data: Additional domain-specific data to save
            
        Returns:
            Path to saved checkpoint file
        """
        
        # Check if this is a regular checkpoint epoch (co 10 epok)
        if epoch % self.checkpoint_frequency == 0:
            return self._save_regular_checkpoint(
                generator, discriminator, optimizer_g, optimizer_d,
                epoch, iteration, generator_loss, discriminator_loss,
                **domain_specific_data
            )
        
        # Check if this is a new best model
        if generator_loss < self.best_generator_loss:
            self.best_generator_loss = generator_loss
            return self._save_best_checkpoint(
                generator, discriminator, optimizer_g, optimizer_d,
                epoch, iteration, generator_loss, discriminator_loss,
                **domain_specific_data
            )
        
        return ""  # No checkpoint saved
    
    def _save_regular_checkpoint(self, generator, discriminator,
                                optimizer_g, optimizer_d,
                                epoch: int, iteration: int,
                                generator_loss: float, discriminator_loss: float,
                                **domain_specific_data) -> str:
        """Save regular checkpoint with rolling buffer strategy"""
        
        # Create checkpoint filename
        filename = f"checkpoint_epoch_{epoch}.tar"
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        
        # Save checkpoint content
        self._save_checkpoint_content(
            checkpoint_path, generator, discriminator, optimizer_g, optimizer_d,
            epoch, iteration, generator_loss, discriminator_loss,
            "regular", **domain_specific_data
        )
        
        # Add to history and manage rolling buffer
        self.checkpoint_history.append(checkpoint_path)
        
        # Auto-cleanup: remove oldest if exceeding buffer size
        if len(self.checkpoint_history) > self.rolling_buffer_size:
            oldest_checkpoint = self.checkpoint_history.pop(0)
            if os.path.exists(oldest_checkpoint):
                os.remove(oldest_checkpoint)
                print(f"🧹 Removed old checkpoint: {os.path.basename(oldest_checkpoint)}")
        
        print(f"💾 Regular checkpoint saved: {filename} (epoch {epoch})")
        return checkpoint_path
    
    def _save_best_checkpoint(self, generator, discriminator,
                             optimizer_g, optimizer_d,
                             epoch: int, iteration: int,
                             generator_loss: float, discriminator_loss: float,
                             **domain_specific_data) -> str:
        """Save best model checkpoint (nie liczy się do rolling buffer)"""
        
        filename = "best_model.tar"
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        
        self._save_checkpoint_content(
            checkpoint_path, generator, discriminator, optimizer_g, optimizer_d,
            epoch, iteration, generator_loss, discriminator_loss,
            "best", **domain_specific_data
        )
        
        print(f"🏆 Best model checkpoint saved: {filename} (loss: {generator_loss:.4f})")
        return checkpoint_path
    
    def save_final_checkpoint(self, generator, discriminator,
                             optimizer_g, optimizer_d,
                             epoch: int, iteration: int,
                             generator_loss: float, discriminator_loss: float,
                             **domain_specific_data) -> str:
        """Save final model checkpoint (końcowy stan - nie liczy się do bufora)"""
        
        filename = "final_model.tar"
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        
        self._save_checkpoint_content(
            checkpoint_path, generator, discriminator, optimizer_g, optimizer_d,
            epoch, iteration, generator_loss, discriminator_loss,
            "final", **domain_specific_data
        )
        
        print(f"🏁 Final model checkpoint saved: {filename}")
        return checkpoint_path
    
    def _save_checkpoint_content(self, checkpoint_path: str, 
                                generator, discriminator, optimizer_g, optimizer_d,
                                epoch: int, iteration: int,
                                generator_loss: float, discriminator_loss: float,
                                checkpoint_type: str, **domain_specific_data):
        """
        Save checkpoint content according to metrics-checkpoints-how-to.txt
        
        ZAWARTOŚĆ KAŻDEGO CHECKPOINTU:
        - model states (generator.pt, discriminator.pt)
        - optimizer states (g_opt.pt, d_opt.pt) 
        - training_context.json (epoch, losses, config, scheduler states)
        - rng_states.pt (torch/numpy/python random states)
        - 5 sample pliki audio/image (sample_1.wav/png, sample_2.wav/png, ...)
        """
        
        # Initialize variables for cleanup
        import shutil
        project_temp_dir = None
        
        try:
            # Create temporary directory for checkpoint content - use project temp dir
            project_temp_dir = os.path.join(self.checkpoint_dir, "temp_checkpoint")
            os.makedirs(project_temp_dir, exist_ok=True)
            
            # Clean up old temp files
            if os.path.exists(project_temp_dir):
                shutil.rmtree(project_temp_dir)
            os.makedirs(project_temp_dir, exist_ok=True)
            
            # Save model states
            generator_path = os.path.join(project_temp_dir, "generator.pt")
            discriminator_path = os.path.join(project_temp_dir, "discriminator.pt")
            torch.save(generator.state_dict(), generator_path)
            torch.save(discriminator.state_dict(), discriminator_path)
            
            # Save optimizer states
            g_opt_path = os.path.join(project_temp_dir, "g_opt.pt")
            d_opt_path = os.path.join(project_temp_dir, "d_opt.pt")
            torch.save(optimizer_g.state_dict(), g_opt_path)
            torch.save(optimizer_d.state_dict(), d_opt_path)
                
                # Save training context
            # Save training context
            training_context = {
                'epoch': epoch,
                'iteration': iteration,
                'generator_loss': generator_loss,
                'discriminator_loss': discriminator_loss,
                'model_type': self.model_type,
                'checkpoint_type': checkpoint_type,
                'timestamp': datetime.now().isoformat(),
                **domain_specific_data
            }
            
            context_path = os.path.join(project_temp_dir, "training_context.json")
            with open(context_path, 'w') as f:
                json.dump(training_context, f, indent=2, default=str)
            
            # Save RNG states
            rng_states = {
                'python_random_state': random.getstate(),
                'numpy_random_state': np.random.get_state(),
                'torch_random_state': torch.get_rng_state()
            }
            
            if torch.cuda.is_available():
                rng_states['torch_cuda_random_state'] = torch.cuda.get_rng_state()
            
            rng_path = os.path.join(project_temp_dir, "rng_states.pt")
            torch.save(rng_states, rng_path)
            
            # Generate 5 sample files (domain-specific)
            sample_paths = self._generate_checkpoint_samples(
                generator, epoch, iteration, checkpoint_type, project_temp_dir
            )
                
            # Create compressed tar archive
            with tarfile.open(checkpoint_path, 'w:gz') as tar:
                    tar.add(generator_path, arcname="generator.pt")
                    tar.add(discriminator_path, arcname="discriminator.pt")
                    tar.add(g_opt_path, arcname="g_opt.pt")
                    tar.add(d_opt_path, arcname="d_opt.pt")
                    tar.add(context_path, arcname="training_context.json")
                    tar.add(rng_path, arcname="rng_states.pt")
                    
                    # Add sample files
                    for i, sample_path in enumerate(sample_paths, 1):
                        if os.path.exists(sample_path):
                            sample_ext = os.path.splitext(sample_path)[1]
                            tar.add(sample_path, arcname=f"sample_{i}{sample_ext}")
            
            # Verify checkpoint was created successfully
            if os.path.exists(checkpoint_path):
                size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
                print(f"✅ Checkpoint content saved: {os.path.basename(checkpoint_path)} ({size_mb:.1f}MB)")
            else:
                raise Exception("Checkpoint file was not created")
                
        except Exception as e:
            print(f"❌ Error saving checkpoint content: {e}")
            raise e
        finally:
            # Clean up project temp directory
            if project_temp_dir and os.path.exists(project_temp_dir):
                shutil.rmtree(project_temp_dir)
    
    def load_checkpoint(self, checkpoint_path: str, generator, discriminator,
                       optimizer_g=None, optimizer_d=None, 
                       load_optimizers: bool = True) -> Dict[str, Any]:
        """
        Load checkpoint and restore model states with new tar.gz structure
        
        Args:
            checkpoint_path: Path to checkpoint file (.tar.gz)
            generator: Generator model to load weights into
            discriminator: Discriminator model to load weights into
            optimizer_g: Generator optimizer (optional)
            optimizer_d: Discriminator optimizer (optional)
            load_optimizers: Whether to load optimizer states
            
        Returns:
            Dictionary with checkpoint metadata
        """
        
        try:
            import tempfile
            
            # Extract tar.gz checkpoint to temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                with tarfile.open(checkpoint_path, 'r:gz') as tar:
                    tar.extractall(temp_dir)
                
                # Load model states
                generator_path = os.path.join(temp_dir, "generator.pt")
                discriminator_path = os.path.join(temp_dir, "discriminator.pt")
                
                if os.path.exists(generator_path):
                    generator.load_state_dict(torch.load(generator_path, map_location='cpu'))
                if os.path.exists(discriminator_path):
                    discriminator.load_state_dict(torch.load(discriminator_path, map_location='cpu'))
                
                # Load optimizer states if requested
                if load_optimizers:
                    g_opt_path = os.path.join(temp_dir, "g_opt.pt")
                    d_opt_path = os.path.join(temp_dir, "d_opt.pt")
                    
                    if optimizer_g is not None and os.path.exists(g_opt_path):
                        optimizer_g.load_state_dict(torch.load(g_opt_path, map_location='cpu'))
                    if optimizer_d is not None and os.path.exists(d_opt_path):
                        optimizer_d.load_state_dict(torch.load(d_opt_path, map_location='cpu'))
                
                # Load training context
                context_path = os.path.join(temp_dir, "training_context.json")
                training_context = {}
                if os.path.exists(context_path):
                    with open(context_path, 'r') as f:
                        training_context = json.load(f)
                
                # Restore RNG states for reproducibility
                rng_path = os.path.join(temp_dir, "rng_states.pt")
                if os.path.exists(rng_path):
                    rng_states = torch.load(rng_path, map_location='cpu')
                    
                    if 'python_random_state' in rng_states:
                        random.setstate(rng_states['python_random_state'])
                    if 'numpy_random_state' in rng_states:
                        np.random.set_state(rng_states['numpy_random_state'])
                    if 'torch_random_state' in rng_states:
                        torch.set_rng_state(rng_states['torch_random_state'])
                    if 'torch_cuda_random_state' in rng_states and torch.cuda.is_available():
                        torch.cuda.set_rng_state(rng_states['torch_cuda_random_state'])
            
            print(f"✅ Checkpoint loaded successfully: {os.path.basename(checkpoint_path)}")
            
            return {
                'epoch': training_context.get('epoch', 0),
                'iteration': training_context.get('iteration', 0),
                'generator_loss': training_context.get('generator_loss', float('inf')),
                'discriminator_loss': training_context.get('discriminator_loss', float('inf')),
                'model_type': training_context.get('model_type', self.model_type),
                'checkpoint_type': training_context.get('checkpoint_type', 'unknown'),
                'timestamp': training_context.get('timestamp', ''),
                **{k: v for k, v in training_context.items() 
                   if k not in ['epoch', 'iteration', 'generator_loss', 'discriminator_loss', 
                               'model_type', 'checkpoint_type', 'timestamp']}
            }
            
        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")
            return {}
    
    def find_latest_checkpoint(self, checkpoint_type: str = "any") -> Optional[str]:
        """
        Find the most recent checkpoint according to optimized structure
        
        Args:
            checkpoint_type: Type of checkpoint to find (regular, best, final, any)
            
        Returns:
            Path to latest checkpoint or None if not found
        """
        
        if checkpoint_type == "any":
            # Search all checkpoint types in priority order
            search_patterns = [
                "final_model.tar",
                "best_model.tar", 
                "checkpoint_epoch_*.tar"
            ]
        elif checkpoint_type == "regular":
            search_patterns = ["checkpoint_epoch_*.tar"]
        elif checkpoint_type == "best":
            search_patterns = ["best_model.tar"]
        elif checkpoint_type == "final":
            search_patterns = ["final_model.tar"]
        else:
            search_patterns = [f"{checkpoint_type}.tar"]
        
        latest_checkpoint = None
        latest_time = 0
        
        for pattern in search_patterns:
            if "*" in pattern:
                # Handle wildcard patterns
                import glob
                pattern_path = os.path.join(self.checkpoint_dir, pattern)
                for file_path in glob.glob(pattern_path):
                    if os.path.isfile(file_path):
                        file_time = os.path.getmtime(file_path)
                        if file_time > latest_time:
                            latest_time = file_time
                            latest_checkpoint = file_path
            else:
                # Handle exact filenames
                file_path = os.path.join(self.checkpoint_dir, pattern)
                if os.path.exists(file_path):
                    file_time = os.path.getmtime(file_path)
                    if file_time > latest_time:
                        latest_time = file_time
                        latest_checkpoint = file_path
        
        if latest_checkpoint:
            print(f"🔍 Latest checkpoint found: {os.path.basename(latest_checkpoint)}")
        
        return latest_checkpoint
    
    def get_checkpoint_info(self) -> Dict[str, Any]:
        """
        Get information about available checkpoints
        
        Returns:
            Dictionary with checkpoint information
        """
        info = {
            'total_checkpoints': 0,
            'regular_checkpoints': [],
            'best_checkpoint': None,
            'final_checkpoint': None,
            'latest_checkpoint': None,
            'disk_usage_mb': 0
        }
        
        if not os.path.exists(self.checkpoint_dir):
            return info
        
        # Scan checkpoint directory
        for filename in os.listdir(self.checkpoint_dir):
            file_path = os.path.join(self.checkpoint_dir, filename)
            
            if not filename.endswith('.tar'):
                continue
                
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            info['disk_usage_mb'] += file_size
            info['total_checkpoints'] += 1
            
            if filename.startswith('checkpoint_epoch_'):
                info['regular_checkpoints'].append({
                    'path': file_path,
                    'filename': filename,
                    'size_mb': file_size,
                    'modified': datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
                })
            elif filename == 'best_model.tar':
                info['best_checkpoint'] = {
                    'path': file_path,
                    'size_mb': file_size,
                    'modified': datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
                }
            elif filename == 'final_model.tar':
                info['final_checkpoint'] = {
                    'path': file_path,
                    'size_mb': file_size,
                    'modified': datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
                }
        
        # Sort regular checkpoints by epoch number
        info['regular_checkpoints'].sort(key=lambda x: x['filename'])
        
        # Find latest checkpoint
        info['latest_checkpoint'] = self.find_latest_checkpoint("any")
        
        return info
    
    def resume_training_info(self) -> Dict[str, Any]:
        """
        Get information needed to resume training from the latest checkpoint
        
        Returns:
            Dictionary with resume information
        """
        latest_checkpoint = self.find_latest_checkpoint("any")
        
        if not latest_checkpoint:
            return {'can_resume': False, 'message': 'No checkpoints found'}
        
        try:
            # Load training context from checkpoint
            import tempfile
            with tempfile.TemporaryDirectory() as temp_dir:
                with tarfile.open(latest_checkpoint, 'r:gz') as tar:
                    tar.extractall(temp_dir)
                
                context_path = os.path.join(temp_dir, "training_context.json")
                if os.path.exists(context_path):
                    with open(context_path, 'r') as f:
                        context = json.load(f)
                    
                    return {
                        'can_resume': True,
                        'checkpoint_path': latest_checkpoint,
                        'last_epoch': context.get('epoch', 0),
                        'last_iteration': context.get('iteration', 0),
                        'last_generator_loss': context.get('generator_loss', float('inf')),
                        'last_discriminator_loss': context.get('discriminator_loss', float('inf')),
                        'checkpoint_type': context.get('checkpoint_type', 'unknown'),
                        'timestamp': context.get('timestamp', ''),
                        'model_type': context.get('model_type', self.model_type),
                        'message': f"Can resume from epoch {context.get('epoch', 0)}"
                    }
        except Exception as e:
            return {'can_resume': False, 'message': f'Error reading checkpoint: {e}'}
        
        return {'can_resume': False, 'message': 'Invalid checkpoint format'}
        for filename in os.listdir(directory):
            if filename.endswith(('.tar', '.tar.gz')):
                file_path = os.path.join(directory, filename)
                files.append((file_path, os.path.getmtime(file_path)))
        
        # Sort by modification time (newest first)
        files.sort(key=lambda x: x[1], reverse=True)
        
        # Remove old files beyond keep_n
        for file_path, _ in files[keep_n:]:
            try:
                os.remove(file_path)
                print(f"🗑️  Removed old checkpoint: {os.path.basename(file_path)}")
            except Exception as e:
                print(f"⚠️  Error removing {file_path}: {e}")
    
    def _compress_checkpoint(self, checkpoint_path: str):
        """Compress a checkpoint file to save disk space"""
        try:
            compressed_path = checkpoint_path + '.gz'
            with open(checkpoint_path, 'rb') as f_in:
                with tarfile.open(compressed_path, 'w:gz') as tar:
                    tarinfo = tarfile.TarInfo(os.path.basename(checkpoint_path))
                    tarinfo.size = os.path.getsize(checkpoint_path)
                    tar.addfile(tarinfo, f_in)
            
            # Remove original uncompressed file
            os.remove(checkpoint_path)
            print(f"📦 Checkpoint compressed: {os.path.basename(compressed_path)}")
            
        except Exception as e:
            print(f"⚠️  Error compressing checkpoint: {e}")
    
    def _decompress_checkpoint(self, compressed_path: str) -> str:
        """Decompress a checkpoint file"""
        try:
            # Extract to temporary location
            temp_path = compressed_path.replace('.tar.gz', '.tar')
            with tarfile.open(compressed_path, 'r:gz') as tar:
                tar.extractall(path=os.path.dirname(temp_path))
            
            return temp_path
            
        except Exception as e:
            print(f"⚠️  Error decompressing checkpoint: {e}")
            return compressed_path
    
    @abstractmethod
    def _generate_checkpoint_samples(self, generator, epoch: int, iteration: int,
                                   checkpoint_type: str, output_dir: Optional[str] = None) -> List[str]:
        """
        Generate 5 sample files for checkpoint
        
        Args:
            generator: Generator model
            epoch: Current epoch
            iteration: Current iteration  
            checkpoint_type: Type of checkpoint
            output_dir: Directory to save samples (if None, uses default)
            
        Returns:
            List of paths to generated sample files
        """
        pass
    
    def is_best_checkpoint(self, current_loss: float, 
                          improvement_threshold: float = 0.05) -> bool:
        """Check if current model is the best so far
        
        Args:
            current_loss: Current generator loss
            improvement_threshold: Minimum improvement threshold (5% by default)
            
        Returns:
            True if this is the best model so far
        """
        
        improvement = (self.best_generator_loss - current_loss) / self.best_generator_loss
        
        if improvement > improvement_threshold:
            self.best_generator_loss = current_loss
            return True
            
        return False
    
    def get_checkpoint_stats(self) -> Dict[str, Any]:
        """Get statistics about saved checkpoints"""
        stats = {
            'total_checkpoints': 0,
            'emergency_checkpoints': 0,
            'milestone_checkpoints': 0,
            'epoch_checkpoints': 0,
            'best_checkpoints': 0,
            'final_checkpoints': 0,
            'total_size_mb': 0.0
        }
        
        for subdir in ["emergency", "milestones", "key_epochs", "best", "final"]:
            dir_path = os.path.join(self.checkpoint_dir, subdir)
            if os.path.exists(dir_path):
                count = 0
                size = 0
                for filename in os.listdir(dir_path):
                    if filename.endswith(('.tar', '.tar.gz')):
                        count += 1
                        file_path = os.path.join(dir_path, filename)
                        size += os.path.getsize(file_path)
                
                stats[f'{subdir}_checkpoints'] = count
                stats['total_checkpoints'] += count
                stats['total_size_mb'] += size / (1024 * 1024)
        
        return stats
