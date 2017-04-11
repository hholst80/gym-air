from gym.envs.registration import register

for game in ['air_raid', 'alien', 'amidar', 'assault', 'asterix', 'asteroids',
             'atlantis', 'bank_heist', 'battle_zone', 'beam_rider', 'berzerk',
             'bowling', 'boxing', 'breakout', 'carnival', 'centipede',
             'chopper_command', 'crazy_climber', 'demon_attack', 'double_dunk',
             'elevator_action', 'enduro', 'fishing_derby', 'freeway',
             'frostbite', 'gopher', 'gravitar', 'ice_hockey', 'jamesbond',
             'journey_escape', 'kangaroo', 'krull', 'kung_fu_master',
             'montezuma_revenge', 'ms_pacman', 'name_this_game', 'phoenix',
             'pitfall', 'pong', 'pooyan', 'private_eye', 'qbert', 'riverraid',
             'road_runner', 'robotank', 'seaquest', 'skiing', 'solaris',
             'space_invaders', 'star_gunner', 'tennis', 'time_pilot',
             'tutankham', 'up_n_down', 'venture', 'video_pinball',
             'wizard_of_wor', 'yars_revenge', 'zaxxon']:

        name = ''.join([g.capitalize() for g in game.split('_')])

        if game == 'space_invaders':
            frameskip = 3
        else:
            frameskip = 4

        nondeterministic = (game == 'elevator_action')

        id = '{name}-image-ram-v0'.format(name=name)

        register(
            id=id,
            entry_point='gym_air.envs:AIREnv',
            kwargs={'game': game, 'frameskip': frameskip,
                    'repeat_action_probability': 0},
            tags={'wrapper_config.TimeLimit.max_episode_steps': 100000},
            nondeterministic=nondeterministic,
        )

        id = '{name}-image-only-v0'.format(name=name)

        register(
            id=id,
            entry_point='gym_air.envs:AIREnv',
            kwargs={'game': game, 'frameskip': frameskip,
                    'use_tuple': False,
                    'repeat_action_probability': 0},
            tags={'wrapper_config.TimeLimit.max_episode_steps': 100000},
            nondeterministic=nondeterministic,
        )
