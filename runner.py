"""
Batch runner for all Dallas County scrapers.
Designed for Google Cloud Run Jobs — runs once and exits.

Environment variables:
  Required:
    SUPABASE_URL
    SUPABASE_SERVICE_ROLE_KEY

  Optional — scraper selection (set to "true" to skip):
    SKIP_TAX_DELINQUENT
    SKIP_FORECLOSURE
    SKIP_PROBATE
    SKIP_DIVORCE

  Optional — tax delinquent config:
    MAX_ACCOUNTS          (default: 50000)
    MAILING_ZIPS          (default: "")
    BATCH_SIZE            (default: 500)
    EXPORT_MODE           (default: "both")

  Optional — foreclosure config:
    FORECLOSURE_MAX_PDFS  (default: 1)

  Optional — probate config:
    PROBATE_LOOKBACK_DAYS (default: 30)

  Optional — divorce config:
    DIVORCE_LOOKBACK_DAYS (default: 30)
    DIVORCE_FETCH_DETAILS (default: "false")

  Optional — GCS:
    GCS_BUCKET
"""

import os
import traceback

import divorce
import foreclosure
import main
import probate


def _skip(name: str) -> bool:
    return os.getenv(f"SKIP_{name.upper()}", "").lower() == "true"


def main_runner():
    batch_size  = int(os.getenv("BATCH_SIZE", "500"))
    export_mode = os.getenv("EXPORT_MODE", "both")
    debug       = os.getenv("DEBUG", "").lower() == "true"

    results = {}

    # ── Tax Delinquent ────────────────────────────────────────────────────────
    if not _skip("tax_delinquent"):
        print("[runner] Starting: tax_delinquent")
        try:
            res = main.run(
                debug=debug,
                max_accounts=int(os.getenv("MAX_ACCOUNTS", "50000")),
                mailing_zips=os.getenv("MAILING_ZIPS", ""),
                batch_size=batch_size,
                export_mode=export_mode,
            )
            results["tax_delinquent"] = res
        except Exception:
            results["tax_delinquent"] = {"ok": False, "error": traceback.format_exc()}
        print(f"[runner] Done: tax_delinquent → leads={results['tax_delinquent'].get('leads_upserted', 'err')}")
    else:
        print("[runner] Skipping: tax_delinquent")

    # ── Foreclosure ───────────────────────────────────────────────────────────
    if not _skip("foreclosure"):
        print("[runner] Starting: foreclosure")
        try:
            res = foreclosure.run(
                batch_size=batch_size,
                export_mode=export_mode,
                debug=debug,
                max_pdfs=int(os.getenv("FORECLOSURE_MAX_PDFS", "1")),
            )
            results["foreclosure"] = res
        except Exception:
            results["foreclosure"] = {"ok": False, "error": traceback.format_exc()}
        print(f"[runner] Done: foreclosure → leads={results['foreclosure'].get('leads_upserted', 'err')}")
    else:
        print("[runner] Skipping: foreclosure")

    # ── Probate ───────────────────────────────────────────────────────────────
    if not _skip("probate"):
        print("[runner] Starting: probate")
        try:
            res = probate.run(
                batch_size=batch_size,
                export_mode=export_mode,
                debug=debug,
                lookback_days=int(os.getenv("PROBATE_LOOKBACK_DAYS", "30")),
            )
            results["probate"] = res
        except Exception:
            results["probate"] = {"ok": False, "error": traceback.format_exc()}
        print(f"[runner] Done: probate → leads={results['probate'].get('leads_upserted', 'err')}")
    else:
        print("[runner] Skipping: probate")

    # ── Divorce ───────────────────────────────────────────────────────────────
    if not _skip("divorce"):
        print("[runner] Starting: divorce")
        try:
            res = divorce.run(
                batch_size=batch_size,
                export_mode=export_mode,
                debug=debug,
                lookback_days=int(os.getenv("DIVORCE_LOOKBACK_DAYS", "30")),
                fetch_details=os.getenv("DIVORCE_FETCH_DETAILS", "").lower() == "true",
            )
            results["divorce"] = res
        except Exception:
            results["divorce"] = {"ok": False, "error": traceback.format_exc()}
        print(f"[runner] Done: divorce → leads={results['divorce'].get('leads_upserted', 'err')}")
    else:
        print("[runner] Skipping: divorce")

    print("\n[runner] ── Summary ──")
    for name, res in results.items():
        ok = res.get("ok", False)
        leads = res.get("leads_upserted", "n/a")
        err = res.get("error", "")
        status = "✓" if ok else "✗"
        print(f"  {status} {name}: leads={leads}" + (f"  ERROR={err[:120]}" if err else ""))

    print("\nRUN_RESULTS:", results)


if __name__ == "__main__":
    main_runner()
