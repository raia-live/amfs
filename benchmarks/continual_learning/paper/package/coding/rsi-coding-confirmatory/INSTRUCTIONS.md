# Coding instructions

Each row is a rule an automated system wrote for a support or CI agent after that agent kept failing a class of task. Read the text alone and answer one question:

**Does the rule state a condition — an input-dependent branch — under which what the agent should do differs?** `Y` if the text names something about the case (a plan, a platform, a directory touched, a keyword, an amount, a customer attribute) that decides between two different actions; `N` if it prescribes one action regardless (an unconditional rule, or one whose only condition is the class itself, e.g. "for Android notification tickets, do X").

Put `Y` or `N` in `conditional (Y/N)`. If `Y`, paste the phrase that states the condition into `condition_quoted`. Do not discuss items with the other coder. Do not look up the tasks.
