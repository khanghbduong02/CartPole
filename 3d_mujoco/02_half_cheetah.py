import os
import time
import warnings
import gymnasium as gym
from stable_baselines3 import PPO

# Suppress the Stable-Baselines3 GPU policy warning
warnings.filterwarnings("ignore", category=UserWarning, module="stable_baselines3")

def main():
    # 1. Configuration Setup
    env_id = "HalfCheetah-v5"
    TOTAL_STEPS = 2000000  
    
    # PERMANENT PATH FIX: Find this script's directory, then point up one level to 'models'
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.abspath(os.path.join(script_dir, "..", "models"))
    os.makedirs(models_dir, exist_ok=True)
    model_save_path = os.path.join(models_dir, "ppo_mujoco_cheetah")
    
    # 2. Train the Model on CUDA (GPU)
    print(f"Initializing 3D MuJoCo {env_id} environment...")
    train_env = gym.make(env_id)
    
    print("Setting up PPO algorithm on CUDA with exploration tweaks...")
    model = PPO(
        "MlpPolicy", 
        train_env, 
        verbose=1, 
        learning_rate=0.0001, 
        ent_coef=0.01, 
        device="cuda"
    )
    
    print(f"Training the cheetah for {TOTAL_STEPS:,} steps. Please wait...")
    model.learn(total_timesteps=TOTAL_STEPS)
    
    # Save model weights to the central directory
    model.save(model_save_path)
    train_env.close()
    print(f"Training complete! Model saved to '{model_save_path}.zip'.")

    # 3. Watch the Trained Cheetah Run Natively
    print("\nLaunching 3D window to watch the trained agent play...")
    test_env = gym.make(env_id, render_mode="human")
    model = PPO.load(model_save_path, env=test_env, device="cuda")
    
    obs, _ = test_env.reset()
    try:
        for episode in range(3):
            print(f"Starting Visual Episode {episode + 1}")
            terminated, truncated = False, False
            step_count = 0
            
            while not (terminated or truncated):
                action, _states = model.predict(obs, deterministic=True)
                
                try:
                    obs, reward, terminated, truncated, info = test_env.step(action)
                except Exception:
                    print("Window manually closed or ESC pressed. Exiting cleanly...")
                    return
                    
                step_count += 1
                time.sleep(1.0 / 60.0) 
                
                if terminated or truncated:
                    print(f"-> Episode finished at step {step_count}! Freezing screen for 1.5 seconds...")
                    time.sleep(1.5)
                
            obs, _ = test_env.reset()
    finally:
        try:
            test_env.close()
        except Exception:
            pass
            
    print("Simulation finished successfully!")

if __name__ == "__main__":
    main()
