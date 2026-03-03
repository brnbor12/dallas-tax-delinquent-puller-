"""Celery tasks for email/webhook alerts on saved searches."""

from __future__ import annotations

import logging

from tasks.celery_app import app

logger = logging.getLogger(__name__)


@app.task
def send_alert(saved_search_id: int):
    """Check a saved search for new matches and send alert if found."""
    # TODO Phase 3: implement email/Slack notifications
    logger.info("send_alert_stub", saved_search_id=saved_search_id)
