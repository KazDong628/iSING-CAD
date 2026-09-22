from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from .config import Settings, load_local_env

def main():
    load_local_env()
    parser = argparse.ArgumentParser(prog="contour-agent")
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve", help="Open the local drawing workbench")
    serve.add_argument("--port", type=int, default=int(os.getenv("CONTOUR_PORT", "8769")))
    catalog = sub.add_parser("catalog", help="Read-only dataset audit")
    catalog.add_argument("--output", type=Path)
    evaluate = sub.add_parser("evaluate", help="Run declared online/offline qualification")
    evaluate.add_argument("--online", action="store_true")
    evaluate.add_argument("--repeats", type=int, default=1)
    evaluate.add_argument("--output", type=Path)
    evaluate.add_argument("--engine", choices=("autonomous", "template"), default="autonomous")
    evaluate.add_argument("--segmentation", action="store_true", help="Evaluate configured segmentation checkpoint")
    evaluate.add_argument("--cases", nargs="+", help="Optional case IDs; the full catalog remains in the denominator")
    args = parser.parse_args()
    settings = Settings()
    if args.command == "serve":
        import uvicorn
        from .server import create_app
        uvicorn.run(create_app(settings), host="127.0.0.1", port=args.port, access_log=False)
    elif args.command == "catalog":
        from .dataset import build_catalog
        data = build_catalog(settings.dataset_root)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(data["counts"], ensure_ascii=False))
    else:
        if not 1 <= args.repeats <= 5:
            parser.error("--repeats must be between 1 and 5")
        if args.engine == "autonomous":
            from .autonomous_qualification import qualify_autonomous
            report, path = qualify_autonomous(settings, online=args.online, repeats=args.repeats, cases=args.cases, use_segmentation=args.segmentation)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf8")
            print(json.dumps({"report": str(path), "qualification": report["qualification"], "summary": report["summary"]}, ensure_ascii=False, indent=2))
            passed = report["qualification"]["strict_requested_scope_passed"]
        else:
            if args.segmentation:
                parser.error("--segmentation requires --engine autonomous")
            if args.cases:
                parser.error("--cases is only available for the autonomous engine")
            from .qualification import qualify
            report, path = qualify(settings, online=args.online, repeats=args.repeats, output=args.output)
            print(json.dumps({"report": str(path), "qualification": report["qualification"], "coverage": report["coverage"]}, ensure_ascii=False, indent=2))
            passed = report["qualification"]["online_assisted_scope_passed"] if args.online else True
        if not passed:
            raise SystemExit(2)

if __name__ == "__main__":
    main()
