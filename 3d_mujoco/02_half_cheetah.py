import os
import time
import gymnasium as gym
from stable_baselines3 import SAC

def main():
    # 1. Configuration Setup
    env_id = "HalfCheetah-v5"
    TOTAL_STEPS = 1000000  # SAC learns much faster; 1M steps is plenty
    
    # Path setup relative to where the file lives
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.abspath(os.path.join(script_dir, "..", "models"))
    os.makedirs(models_dir, exist_ok=True)
    model_save_path = os.path.join(models_dir, "sac_mujoco_cheetah")
    
    # 2. Train the Model on CPU
    print(f"Initializing 3D MuJoCo {env_id} environment...")
    train_env = gym.make(env_id)
    
    print("Setting up SAC (Soft Actor-Critic) algorithm on CPU...")
    # SAC BENCHMARK FOR LOCOMOTION:
    # - buffer_size=300000: Stores experiences to sample from randomly
    # - learning_rate=0.0003: Smooth adjustments to continuous joint forces
    # - ent_coef="auto": The AI automatically tunes its exploration to avoid belly sliding
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
        device="cpu"  # Fast and efficient vector processing on CPU
    )
    
    print(f"Training the cheetah with SAC for {TOTAL_STEPS:,} steps. Please wait...")
    model.learn(total_timesteps=TOTAL_STEPS)
    
    # Save model weights to the central directory
    model.save(model_save_path)
    train_env.close()
    print(f"Training complete! Model saved to '{model_save_path}.zip'.")

    # 3. Watch the Trained Agent Play
    print("\nLaunching 3D window to watch the trained agent play...")
    
    # --- AUTOMATIC GROUND FLOOR EXTENSION CODES ---
    # We load the environment configuration definitions into memory first
    base_test_env = gym.make(env_id, render_mode="human")
    
    # Access the underlying MuJoCo engine model structural descriptors
    # The 'floor' geometry layer is traditionally stored at index 0 of the model geoms
    try:
        # Increase the size of the floor plane array 
        # Changing from standard small values to [700, 700, 1] creates a massive grid area
        base_test_env.unwrapped.model.geom_size[0, :] = [700.0, 700.0, 1.0]
        print("-> Checkered ground floor area expanded successfully to a 700x700 zone!")
    except Exception as e:
        print(f"Note: Could not automatically upscale the floor texture grid: {e}")
    # ----------------------------------------------
    
    test_env = base_test_env # Map your modified env back to the test loop variable
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
