import os
import time
import gymnasium as gym
from stable_baselines3 import PPO

def main():
    # 1. Configuration Setup
    env_id = "InvertedPendulum-v5"
    CUSTOM_STEP_LIMIT = 5000 
    
    # Define and automatically create a central models directory one level up
    models_dir = os.path.join("..", "models")
    os.makedirs(models_dir, exist_ok=True)
    model_save_path = os.path.join(models_dir, "ppo_mujoco_3d_cartpole")
    
    # 2. Train the Model
    print("Initializing modern 3D MuJoCo environment...")
    base_train_env = gym.make(env_id)
    train_env = gym.wrappers.TimeLimit(base_train_env, max_episode_steps=CUSTOM_STEP_LIMIT)
    
    print("Setting up PPO algorithm...")
    # Explicitly using "cpu" for training on the CPU
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
    for episode in range(5):
        print(f"Starting Visual Episode {episode + 1}")
        terminated = False
        truncated = False
        step_count = 0
        
        while not (terminated or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)
            step_count += 1
            
            time.sleep(1.0 / 60.0)
            
            if terminated or truncated:
                if terminated:
                    print(f"-> Pole fell at step {step_count}! Freezing screen for 1.5 seconds...")
                elif truncated:
                    print(f"-> Reached CUSTOM time limit of {CUSTOM_STEP_LIMIT} steps! Freezing screen...")
                
                time.sleep(1.5) 
            
        obs, _ = test_env.reset()
        
    test_env.close()
    print("Simulation finished successfully!")

if __name__ == "__main__":
    main()
