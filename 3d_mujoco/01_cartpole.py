import os
import time
import gymnasium as gym
from stable_baselines3 import PPO

def main():
    # 1. Configuration Setup
    env_id = "InvertedPendulum-v5"
    CUSTOM_STEP_LIMIT = 5000 
    
    # PERMANENT PATH FIX: Find this script's directory, then point up one level to 'models'
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.abspath(os.path.join(script_dir, "..", "models"))
    os.makedirs(models_dir, exist_ok=True)
    model_save_path = os.path.join(models_dir, "ppo_mujoco_3d_cartpole")
    
    # 2. Train the Model (Optimized on CPU)
    print("Initializing modern 3D MuJoCo environment...")
    base_train_env = gym.make(env_id)
    train_env = gym.wrappers.TimeLimit(base_train_env, max_episode_steps=CUSTOM_STEP_LIMIT)
    
    print("Setting up PPO algorithm...")
    model = PPO("MlpPolicy", train_env, verbose=1, learning_rate=0.0003, device="cpu")
    
    print("Training the agent for 200,000 steps. Please wait...")
    model.learn(total_timesteps=200000)
    
    # Save model weights to the central directory
    model.save(model_save_path)
    train_env.close()
    print(f"Training complete! Model saved to '{model_save_path}.zip'.")

    # 3. Watch the Trained Model Play
    print("\nLaunching 3D window to watch the trained agent play...")
    base_test_env = gym.make(env_id, render_mode="human")
    test_env = gym.wrappers.TimeLimit(base_test_env, max_episode_steps=CUSTOM_STEP_LIMIT)
    
    # Load model weights from the central directory
    model = PPO.load(model_save_path, env=test_env, device="cpu")
    
    obs, _ = test_env.reset()
    try:
        for episode in range(5):
            print(f"Starting Visual Episode {episode + 1}")
            terminated, truncated = False, False
            step_count = 0
            
            while not (terminated or truncated):
                action, _states = model.predict(obs, deterministic=True)
                
                # Wrapped step in try-except block to handle emergency exit gracefully
                try:
                    obs, reward, terminated, truncated, info = test_env.step(action)
                except Exception:
                    print("Window manually closed or ESC pressed. Exiting cleanly...")
                    return
                    
                step_count += 1
                time.sleep(1.0 / 60.0)
                
                if terminated or truncated:
                    if terminated:
                        print(f"-> Pole fell at step {step_count}! Freezing screen for 1.5 seconds...")
                    elif truncated:
                        print(f"-> Reached CUSTOM time limit of {CUSTOM_STEP_LIMIT} steps! Freezing screen...")
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
