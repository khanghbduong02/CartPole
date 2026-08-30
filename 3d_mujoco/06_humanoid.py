import os
import time
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback

def main():
    # 1. Configuration Setup
    env_id = "Humanoid-v5"
    TOTAL_STEPS = 3000000  # Humanoids have 17 joints; they need a 3M step budget to stabilize
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.abspath(os.path.join(script_dir, "..", "models"))
    os.makedirs(models_dir, exist_ok=True)
    
    # Absolute path for saving the best walking model weights
    best_model_path = os.path.join(models_dir, "best_sac_mujoco_humanoid")
    
    # 2. Train the Model on CPU
    print(f"Initializing 3D MuJoCo {env_id} environment...")
    train_env = gym.make(env_id)
    eval_env = gym.make(env_id)
    
    # Checkpoint evaluation system to capture the model before any late-stage policy decay
    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path=models_dir,
        log_path=models_dir, 
        eval_freq=40000,  # Checked every 40k steps due to the large action space
        deterministic=True, 
        render=False
    )
    
    print("Setting up stabilized SAC algorithm on CPU for 17-joint bipedal locomotion...")
    # DEEP BIPEDAL WALKING ARCHITECTURE:
    # We maintain deep layer widths and activate state-dependent exploration (gSDE)
    # to give the human skeleton smooth muscle adjustments (prevents rapid spastic foot twitching)
    policy_kwargs = dict(net_arch=dict(pi=[256, 256], qf=[256, 256]))
    
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
        use_sde=True,
        sde_sample_freq=4,
        policy_kwargs=policy_kwargs,
        device="cpu"  # Highly optimized vector processing on CPU
    )
    
    print(f"Training the 3D Humanoid for {TOTAL_STEPS:,} steps. This will take some time...")
    model.learn(total_timesteps=TOTAL_STEPS, callback=eval_callback)
    
    train_env.close()
    eval_env.close()
    
    # Automatically rename the peak-performing file layout cleanly
    generic_saved_file = os.path.join(models_dir, "best_model.zip")
    custom_target_file = f"{best_model_path}.zip"
    
    if os.path.exists(generic_saved_file):
        if os.path.exists(custom_target_file):
            os.remove(custom_target_file)
        os.rename(generic_saved_file, custom_target_file)
        print(f"Success! Best model renamed and saved as: '{custom_target_file}'")
    else:
        print("Warning: Peak model file not found.")

    # 3. Watch the Trained Humanoid Walk Natively
    print("\nLaunching 3D window to watch the trained agent play...")
    base_test_env = gym.make(env_id, render_mode="human")
    
    # --- AUTOMATIC GROUND FLOOR EXTENSION ZONE FIXED ---
    try:
        base_test_env.unwrapped.model.geom_size[0, :] = [1000.0, 1000.0, 1.0]
        print("-> Checkered ground floor area expanded successfully to a 1000x1000 zone!")
    except Exception as e:
        print(f"Note: Could not automatically upscale the floor texture grid: {e}")
    # ----------------------------------------------------
    
    test_env = base_test_env
    model = SAC.load(best_model_path, env=test_env, device="cpu")
    
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
                        print(f"-> Humanoid lost balance and collapsed at step {step_count}! Freezing screen...")
                    elif truncated:
                        print(f"-> Reached maximum 1000 step limit! Flawless stroll! Freezing screen...")
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
