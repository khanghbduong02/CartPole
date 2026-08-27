import os
import time
import gymnasium as gym
from stable_baselines3 import SAC

def main():
    # 1. Configuration Setup
    env_id = "Hopper-v5"
    TOTAL_STEPS = 1000000  # SAC is highly efficient; 1M steps is plenty
    
    # Path setup relative to where this file lives
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.abspath(os.path.join(script_dir, "..", "models"))
    os.makedirs(models_dir, exist_ok=True)
    model_save_path = os.path.join(models_dir, "sac_mujoco_hopper")
    
    # 2. Train the Model on CPU (Fast vector processing)
    print(f"Initializing 3D MuJoCo {env_id} environment...")
    train_env = gym.make(env_id)
    
    print("Setting up SAC (Soft Actor-Critic) algorithm on CPU...")
    # SAC works beautifully for Hopper because the entropy framework prevents 
    # it from getting stuck just standing still or falling over on purpose.
    model = SAC(
        "MlpPolicy", 
        train_env, 
        verbose=1, 
        learning_rate=0.0003, 
        buffer_size=300000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        ent_coef="auto",
        device="cpu"
    )
    
    print(f"Training the hopper with SAC for {TOTAL_STEPS:,} steps. Please wait...")
    model.learn(total_timesteps=TOTAL_STEPS)
    
    # Save model weights to the central directory
    model.save(model_save_path)
    train_env.close()
    print(f"Training complete! Model saved to '{model_save_path}.zip'.")

    # 3. Watch the Trained Hopper Hop Natively
    print("\nLaunching 3D window to watch the trained agent play...")
    test_env = gym.make(env_id, render_mode="human")
    model = SAC.load(model_save_path, env=test_env, device="cpu")
    
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
                time.sleep(1.0 / 60.0)  # Smooth 60 FPS playback
                
                if terminated or truncated:
                    if terminated:
                        print(f"-> Hopper fell or glitched at step {step_count}! Freezing screen...")
                    elif truncated:
                        print(f"-> Reached full 1000 step limit! Perfect run! Freezing screen...")
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
