"""Celery tasks for score calculation v2 — additive model with stacking bonuses."""
from __future__ import annotations
import structlog
from sqlalchemy import text
from tasks.celery_app import app
logger = structlog.get_logger(__name__)

_BULK = text("""
WITH d_weights(t,w) AS (VALUES
  ('foreclosure'::text,40.0),('pre_foreclosure',35.0),('probate',20.0),
  ('tax_delinquent',20.0),('lien',18.0),('eviction',15.0),
  ('code_violation',15.0),('active_listing',10.0),('expired_listing',10.0),
  ('vacant',10.0),('price_reduction',5.0)
),
o_weights(t,w) AS (VALUES
  ('absentee_owner'::text,10.0),('out_of_state_owner',8.0),('no_homestead',7.0)
),
pi_base AS (
  SELECT pi.property_id, pi.indicator_type, pi.filing_date,
    CASE WHEN pi.indicator_type NOT IN ('tax_delinquent','lien') THEN 1.0
         WHEN pi.amount_cents IS NULL THEN 1.0
         WHEN pi.amount_cents>=5000000 THEN 1.40
         WHEN pi.amount_cents>=2000000 THEN 1.25
         WHEN pi.amount_cents>=1000000 THEN 1.15
         WHEN pi.amount_cents>=500000  THEN 1.05
         ELSE 1.0 END AS amult
  FROM property_indicators pi WHERE pi.status='active'
),
bd AS (
  SELECT DISTINCT ON (p.property_id,p.indicator_type)
    p.property_id, p.indicator_type,
    LEAST(d.w*p.amult, d.w*1.4) AS pts, p.filing_date
  FROM pi_base p JOIN d_weights d ON d.t=p.indicator_type
  ORDER BY p.property_id,p.indicator_type,p.filing_date DESC NULLS LAST
),
bo AS (
  SELECT DISTINCT ON (p.property_id,p.indicator_type)
    p.property_id, p.indicator_type, o.w AS pts, p.filing_date
  FROM pi_base p JOIN o_weights o ON o.t=p.indicator_type
  ORDER BY p.property_id,p.indicator_type,p.filing_date DESC NULLS LAST
),
dt AS (
  SELECT property_id,
    LEAST(SUM(pts),60.0) AS dp, COUNT(*) AS dc,
    MAX(filing_date) AS fd,
    jsonb_object_agg(indicator_type,ROUND(pts::numeric,1)) AS jd
  FROM bd GROUP BY property_id
),
ot AS (
  SELECT property_id,
    LEAST(SUM(pts),25.0) AS op, COUNT(*) AS oc,
    MAX(filing_date) AS fd,
    jsonb_object_agg(indicator_type,ROUND(pts::numeric,1)) AS jo
  FROM bo GROUP BY property_id
),
combined AS (
  SELECT
    COALESCE(d.property_id,o.property_id) AS property_id,
    COALESCE(d.dp,0.0) AS dp,
    COALESCE(o.op,0.0) AS op,
    COALESCE(d.dc,0) AS dc,
    COALESCE(o.oc,0) AS oc,
    GREATEST(d.fd,o.fd) AS freshest,
    COALESCE(d.jd,'{}'::jsonb)||COALESCE(o.jo,'{}'::jsonb) AS breakdown,
    COALESCE(d.dc,0)+COALESCE(o.oc,0) AS total_count
  FROM dt d FULL OUTER JOIN ot o ON o.property_id=d.property_id
),
final AS (
  SELECT property_id, total_count,
    LEAST(ROUND((
      dp + op
      + CASE WHEN dc>=3 THEN 20.0 WHEN dc>=2 THEN 10.0 ELSE 0.0 END
      + CASE WHEN oc>=2 THEN 5.0 ELSE 0.0 END
      + CASE WHEN freshest IS NULL THEN 0.0
             WHEN CURRENT_DATE-freshest<=30 THEN 5.0
             WHEN CURRENT_DATE-freshest<=90 THEN 3.0
             WHEN CURRENT_DATE-freshest<=365 THEN 1.0
             ELSE 0.0 END
    )::numeric,2),100.0) AS total_score,
    breakdown
  FROM combined
)
INSERT INTO property_scores(property_id,total_score,indicator_count,score_breakdown,score_tier,last_scored_at)
SELECT property_id,total_score,total_count,breakdown,
  CASE WHEN total_score>=80 THEN 'hot'
       WHEN total_score>=60 THEN 'warm'
       WHEN total_score>=40 THEN 'nurture'
       ELSE 'cold' END,
  NOW()
FROM final
ON CONFLICT(property_id) DO UPDATE SET
  total_score=EXCLUDED.total_score, indicator_count=EXCLUDED.indicator_count,
  score_breakdown=EXCLUDED.score_breakdown, score_tier=EXCLUDED.score_tier,
  last_scored_at=NOW()
""")

_TARGETED = text("""
WITH d_weights(t,w) AS (VALUES
  ('foreclosure'::text,40.0),('pre_foreclosure',35.0),('probate',20.0),
  ('tax_delinquent',20.0),('lien',18.0),('eviction',15.0),
  ('code_violation',15.0),('active_listing',10.0),('expired_listing',10.0),
  ('vacant',10.0),('price_reduction',5.0)
),
o_weights(t,w) AS (VALUES
  ('absentee_owner'::text,10.0),('out_of_state_owner',8.0),('no_homestead',7.0)
),
pi_base AS (
  SELECT pi.property_id, pi.indicator_type, pi.filing_date,
    CASE WHEN pi.indicator_type NOT IN ('tax_delinquent','lien') THEN 1.0
         WHEN pi.amount_cents IS NULL THEN 1.0
         WHEN pi.amount_cents>=5000000 THEN 1.40
         WHEN pi.amount_cents>=2000000 THEN 1.25
         WHEN pi.amount_cents>=1000000 THEN 1.15
         WHEN pi.amount_cents>=500000  THEN 1.05
         ELSE 1.0 END AS amult
  FROM property_indicators pi WHERE pi.status='active'
    AND pi.property_id=ANY(:property_ids)
),
bd AS (
  SELECT DISTINCT ON (p.property_id,p.indicator_type)
    p.property_id, p.indicator_type,
    LEAST(d.w*p.amult, d.w*1.4) AS pts, p.filing_date
  FROM pi_base p JOIN d_weights d ON d.t=p.indicator_type
  ORDER BY p.property_id,p.indicator_type,p.filing_date DESC NULLS LAST
),
bo AS (
  SELECT DISTINCT ON (p.property_id,p.indicator_type)
    p.property_id, p.indicator_type, o.w AS pts, p.filing_date
  FROM pi_base p JOIN o_weights o ON o.t=p.indicator_type
  ORDER BY p.property_id,p.indicator_type,p.filing_date DESC NULLS LAST
),
dt AS (
  SELECT property_id,LEAST(SUM(pts),60.0) AS dp,COUNT(*) AS dc,
    MAX(filing_date) AS fd,jsonb_object_agg(indicator_type,ROUND(pts::numeric,1)) AS jd
  FROM bd GROUP BY property_id
),
ot AS (
  SELECT property_id,LEAST(SUM(pts),25.0) AS op,COUNT(*) AS oc,
    MAX(filing_date) AS fd,jsonb_object_agg(indicator_type,ROUND(pts::numeric,1)) AS jo
  FROM bo GROUP BY property_id
),
combined AS (
  SELECT COALESCE(d.property_id,o.property_id) AS property_id,
    COALESCE(d.dp,0.0) AS dp,COALESCE(o.op,0.0) AS op,
    COALESCE(d.dc,0) AS dc,COALESCE(o.oc,0) AS oc,
    GREATEST(d.fd,o.fd) AS freshest,
    COALESCE(d.jd,'{}'::jsonb)||COALESCE(o.jo,'{}'::jsonb) AS breakdown,
    COALESCE(d.dc,0)+COALESCE(o.oc,0) AS total_count
  FROM dt d FULL OUTER JOIN ot o ON o.property_id=d.property_id
),
final AS (
  SELECT property_id,total_count,
    LEAST(ROUND((dp+op
      +CASE WHEN dc>=3 THEN 20.0 WHEN dc>=2 THEN 10.0 ELSE 0.0 END
      +CASE WHEN oc>=2 THEN 5.0 ELSE 0.0 END
      +CASE WHEN freshest IS NULL THEN 0.0
             WHEN CURRENT_DATE-freshest<=30 THEN 5.0
             WHEN CURRENT_DATE-freshest<=90 THEN 3.0
             WHEN CURRENT_DATE-freshest<=365 THEN 1.0
             ELSE 0.0 END
    )::numeric,2),100.0) AS total_score,
    breakdown
  FROM combined
)
INSERT INTO property_scores(property_id,total_score,indicator_count,score_breakdown,score_tier,last_scored_at)
SELECT property_id,total_score,total_count,breakdown,
  CASE WHEN total_score>=80 THEN 'hot' WHEN total_score>=60 THEN 'warm'
       WHEN total_score>=40 THEN 'nurture' ELSE 'cold' END, NOW()
FROM final
ON CONFLICT(property_id) DO UPDATE SET
  total_score=EXCLUDED.total_score,indicator_count=EXCLUDED.indicator_count,
  score_breakdown=EXCLUDED.score_breakdown,score_tier=EXCLUDED.score_tier,
  last_scored_at=NOW()
""")

@app.task(bind=True, max_retries=2)
def bulk_rescore_all(self):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from app.core.config import settings
    sync_url = settings.database_url.replace("+asyncpg", "+psycopg2")
    engine = create_engine(sync_url)
    try:
        with sessionmaker(bind=engine)() as session:
            result = session.execute(_BULK)
            rows = result.rowcount
            session.commit()
        logger.info("bulk_rescore_all_completed", properties_scored=rows)
        return {"properties_scored": rows}
    except Exception as exc:
        logger.error("bulk_rescore_all_failed", error=str(exc))
        raise self.retry(exc=exc)

@app.task(bind=True, max_retries=2)
def bulk_rescore_properties(self, property_ids: list[int]):
    if not property_ids:
        return {"properties_scored": 0}
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from app.core.config import settings
    sync_url = settings.database_url.replace("+asyncpg", "+psycopg2")
    engine = create_engine(sync_url)
    try:
        with sessionmaker(bind=engine)() as session:
            result = session.execute(_TARGETED, {"property_ids": property_ids})
            rows = result.rowcount
            session.commit()
        logger.info("bulk_rescore_properties_completed", properties_scored=rows)
        return {"properties_scored": rows}
    except Exception as exc:
        logger.error("bulk_rescore_properties_failed", error=str(exc))
        raise self.retry(exc=exc)

@app.task(bind=True, max_retries=3)
def recalculate_property_score(self, property_id: int):
    from sqlalchemy import create_engine, select
    from sqlalchemy.dialects.postgresql import insert as pg_insert
    from sqlalchemy.orm import sessionmaker
    from app.core.config import settings
    from app.models.indicator import PropertyIndicator
    from app.models.score import PropertyScore
    from scoring.engine import score_from_orm
    sync_url = settings.database_url.replace("+asyncpg", "+psycopg2")
    engine = create_engine(sync_url)
    with sessionmaker(bind=engine)() as session:
        indicators = session.execute(
            select(PropertyIndicator).where(PropertyIndicator.property_id == property_id)
        ).scalars().all()
        result = score_from_orm(list(indicators))
        session.execute(pg_insert(PropertyScore).values(
            property_id=property_id, total_score=result["total_score"],
            indicator_count=result["indicator_count"], score_breakdown=result["breakdown"],
            score_tier=result["tier"],
        ).on_conflict_do_update(index_elements=["property_id"], set_={
            "total_score": result["total_score"], "indicator_count": result["indicator_count"],
            "score_breakdown": result["breakdown"], "score_tier": result["tier"],
        }))
        session.commit()
    logger.info("score_recalculated", property_id=property_id, score=result["total_score"])

@app.task
def nightly_score_decay():
    bulk_rescore_all.delay()
