def get_action_value(mdp, state_values, state, action, gamma):
    """ Вычисляет Q(s,a) согласно формуле выше """
    
    # Ваша имплементация ниже
    
    return sum([mdp.get_transition_prob(state, action, sa) * (mdp.get_reward(state, action, sa) + gamma * state_values[sa]) for sa in state_values])
