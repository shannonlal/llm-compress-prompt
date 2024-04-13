def control_context_budget(context, budget):
    """
    Control the context budget to be within the budget limit
    """
    if context is None:
        return None
    if budget is None:
        return context
    if len(context) <= budget:
        return context
    else:
        return context[:budget]