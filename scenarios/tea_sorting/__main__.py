from scenarios.organizer.runner import main
from . import scenario
from .policy import TeaPolicy

main(task_module=scenario, policy_class=TeaPolicy)
