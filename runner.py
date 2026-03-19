import os
import main

def main_runner():
    export_mode = os.getenv("EXPORT_MODE", "both")
    max_accounts = int(os.getenv("MAX_ACCOUNTS", "50000"))
    mailing_zips = os.getenv("MAILING_ZIPS", "")
    batch_size = int(os.getenv("BATCH_SIZE", "500"))

    res = main.run(
        debug=False,
        max_accounts=max_accounts,
        mailing_zips=mailing_zips,
        batch_size=batch_size,
        export_mode=export_mode,
    )
    print("RUN_RESULT:", res)

if __name__ == "__main__":
    main_runner()
