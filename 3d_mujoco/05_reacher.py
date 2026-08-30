import os
import time
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback

def main():
    # 1. Configuration Setup
    env_id = "Reacher-v5"
    TOTAL_STEPS = 1000000  # 1 Million steps gives the joints plenty of precision training
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.abspath(os.path.join(script_dir, "..", "models"))
    os.makedirs(models_dir, exist_ok=True)
    
    # Absolute path for saving the peak-performing tracking layout
    best_model_path = os.path.join(models_dir, "best_sac_mujoco_reacher")
    
    # 2. Train the Model on CPU
    print(f"Initializing 3D MuJoCo {env_id} environment...")
    train_env = gym.make(env_id)
    eval_env = gym.make(env_id)
    
    # Checkpoint evaluation system to capture the highest precision accuracy weights
    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path=models_dir,
        log_path=models_dir, 
        eval_freq=20000,
        deterministic=True, 
        render=False
    )
    
    print("Setting up SAC algorithm on CPU for target coordinate tracking...")
    # REACHER ACCURACY ARCHITECTURE:
    # We maintain a deep network and activate smooth state-dependent noise (gSDE)
    # to stop the robotic arm from frantically shaking or twitching when it nears the goal.
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
        device="cpu"
    )
    
    print(f"Training the robotic reacher arm for {TOTAL_STEPS:,} steps. Please wait...")
    model.learn(total_timesteps=TOTAL_STEPS, callback=eval_callback)
    
    train_env.close()
    eval_env.close()
    
    # Automatically handle unique filename organization
    generic_saved_file = os.path.join(models_dir, "best_model.zip")
    custom_target_file = f"{best_model_path}.zip"
    
    if os.path.exists(generic_saved_file):
        if os.path.exists(custom_target_file):
            os.remove(custom_target_file)
        os.rename(generic_saved_file, custom_target_file)
        print(f"Success! Best model renamed and saved as: '{custom_target_file}'")
    else:
        print("Warning: Peak model file not found.")

    # 3. Watch the Trained Robotic Arm Track Targets
    print("\nLaunching 3D window to watch the trained agent play...")
    test_env = gym.make(env_id, render_mode="human")
    
    # Load your cleanly isolated peak custom target
    model = SAC.load(best_model_path, env=test_env, device="cpu")
    
    obs, _ = test_env.reset()
    try:
        for episode in range(5):
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
                    # In Reacher, there are no structural failure falls. It runs for 50 steps per target default.
                    print(f"-> Target tracking sequence finished at step {step_count}! Freezing screen...")
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
