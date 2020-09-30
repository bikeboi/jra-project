from experiment import ex


ex.run(config_updates={'model_type': 'supervised'})
ex.run(config_updates={'model_type': 'supervised'}, named_configs=['unmuted_config'])

