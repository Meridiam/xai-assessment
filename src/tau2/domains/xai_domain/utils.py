from datetime import date, datetime

from tau2.utils.utils import DATA_DIR

XAI_DOMAIN_DATA_DIR = DATA_DIR / "tau2" / "domains" / "xai_domain"
XAI_DOMAIN_DB_PATH = XAI_DOMAIN_DATA_DIR / "db.toml"
XAI_DOMAIN_USER_DB_PATH = XAI_DOMAIN_DATA_DIR / "user_db.toml"
XAI_DOMAIN_MAIN_POLICY_PATH = XAI_DOMAIN_DATA_DIR / "main_policy.md"
XAI_DOMAIN_TECH_SUPPORT_POLICY_MANUAL_PATH = XAI_DOMAIN_DATA_DIR / "tech_support_manual.md"
XAI_DOMAIN_TECH_SUPPORT_POLICY_WORKFLOW_PATH = (
    XAI_DOMAIN_DATA_DIR / "tech_support_workflow.md"
)
XAI_DOMAIN_MAIN_POLICY_SOLO_PATH = XAI_DOMAIN_DATA_DIR / "main_policy_solo.md"
XAI_DOMAIN_TECH_SUPPORT_POLICY_MANUAL_SOLO_PATH = (
    XAI_DOMAIN_DATA_DIR / "tech_support_manual.md"
)
XAI_DOMAIN_TECH_SUPPORT_POLICY_WORKFLOW_SOLO_PATH = (
    XAI_DOMAIN_DATA_DIR / "tech_support_workflow_solo.md"
)
XAI_DOMAIN_TASK_SET_PATH_FULL = XAI_DOMAIN_DATA_DIR / "tasks_full.json"
XAI_DOMAIN_TASK_SET_PATH_SMALL = XAI_DOMAIN_DATA_DIR / "tasks_small.json"
XAI_DOMAIN_TASK_SET_PATH = XAI_DOMAIN_DATA_DIR / "tasks.json"


def get_now() -> datetime:
    # assume now is 2025-02-25 12:08:00
    return datetime(2025, 2, 25, 12, 8, 0)


def get_today() -> date:
    # assume today is 2025-02-25
    return date(2025, 2, 25)
