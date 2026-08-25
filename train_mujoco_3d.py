import os
import time
import gymnasium as gym
from stable_baselines3 import PPO

def main():
    # Use Gymnasium's native 3D Inverted Pendulum environment
    # Note: We append '_Gymnasium_Base' to grab the raw environment layout 
    # so we can cleanly wrap our custom time limits manually.
    env_id = "InvertedPendulum-v5"
    
    # Set your custom test step limit here (Default is 1000)
    CUSTOM_STEP_LIMIT = 5000 
    
    # 1. Train the Model (Increased to 200,000 steps so it actually learns to balance)
    print("Initializing modern 3D MuJoCo environment...")
    base_train_env = gym.make(env_id)
    train_env = gym.wrappers.TimeLimit(base_train_env, max_episode_steps=CUSTOM_STEP_LIMIT)
    
    print("Setting up PPO algorithm...")
    model = PPO("MlpPolicy", train_env, verbose=1, learning_rate=0.0003)
    
    # Increased budget from 50,000 to 200,000 steps so the AI survives the longer limit
    print("Training the agent for 200,000 steps. Please wait...")
    model.learn(total_timesteps=200000)
    
    model.save("ppo_mujoco_3d_cartpole")
    train_env.close()
    print("Training complete! Model saved as 'ppo_mujoco_3d_cartpole.zip'.")

    # 2. Watch the Trained Model Play (With higher step limit)
    print("\nLaunching 3D window to watch the trained agent play...")
    base_test_env = gym.make(env_id, render_mode="human")
    test_env = gym.wrappers.TimeLimit(base_test_env, max_episode_steps=CUSTOM_STEP_LIMIT)
    
    model = PPO.load("ppo_mujoco_3d_cartpole", env=test_env)
    
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
