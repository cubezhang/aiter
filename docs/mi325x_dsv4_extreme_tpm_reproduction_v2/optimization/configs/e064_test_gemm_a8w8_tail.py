25201 op_tests/test_gemm_a8w8.py
    default=[dtypes.d_dtypes["i8"], dtypes.d_dtypes["fp8"]],
    help="""Data type of quantization.
    e.g.: -q fp8""",
)
parser.add_argument(
    "--pad_a",
    type=int,
    default=128,
    help="Pad A on K dimension, stride_a = K + pad_a.",
)
parser.add_argument(
    "-mnk",
    type=dtypes.str2tuple,
    nargs="*",
    default=[
        # qkv_proj
        (1, 1280, 8192),
        (32, 1280, 8192),
        (64, 1280, 8192),
        (128, 1280, 8192),
        (192, 1280, 8192),
        (256, 1280, 8192),
        (320, 1280, 8192),
        (512, 1280, 8192),
        (1024, 1280, 8192),
        (2048, 1280, 8192),
        (4096, 1280, 8192),
        (8192, 1280, 8192),
        (16384, 1280, 8192),
        # attn_out
        (1, 8192, 1024),
        (32, 8192, 1024),
        (64, 8192, 1024),
        (128, 8192, 1024),
        (192, 8192, 1024),
        (256, 8192, 1024),
        (320, 8192, 1024),
        (512, 8192, 1024),
        (1024, 8192, 1024),
        (2048, 8192, 1024),
        (4096, 8192, 1024),
        (8192, 8192, 1024),
        (16384, 8192, 1024),
        # hipmm gelu_bias
        (32, 3072, 768),
        (4096, 3072, 768),
        (8192, 3072, 768),
        # hipmm preshuffle
        (16, 7424, 8192),
        (32, 7424, 8192),
        (48, 7424, 8192),
        (64, 7424, 8192),
        (4096, 7424, 8192),
        (5120, 7424, 8192),
        (8192, 7424, 8192),
    ],
    help="""Shape of mnk.
    e.g. -mnk 1280,8192,1024""",
)

parser.add_argument(
    "--csv",
    type=str,
    default=None,
    help="""CSV file containing M, N, K columns (one shape per row).
    e.g.: --csv shapes.csv""",
)
parser.add_argument(
    "--bpreshuffle-csv",
    type=str,
    default=None,
    dest="bpreshuffle_csv",
    help="""CSV file for bpreshuffle-path shapes (skips gemm_a8w8_CK, runs ASM directly).
    e.g.: --bpreshuffle-csv op_tests/configs/gemm_codegen_gfx_filter_bpreshuffle.csv""",
)
parser.add_argument(
    "-o",
    "--output",
    type=str,
    default=None,
    help="""Directory to save results CSV.
    e.g.: -o results/""",
)
parser.add_argument(
    "--suffix",
    type=str,
    default="results",
    help="""Suffix for output CSV filename.
    e.g.: --suffix branch""",
)
parser.add_argument(
    "--no-flydsl-csv",
    action="store_true",
    help="Skip validating flydsl shapes from tuned bpreshuffle CSVs.",
)
parser.add_argument(
    "--no-legacy",
    action="store_true",
    help="Skip the original hardcoded shape sweep and skinny tests.",
)


args = parser.parse_args()

if not args.no_flydsl_csv:
    bench_csv = os.environ.get("AITER_TUNED_OP_BENCH_CSV", "tuned_op_bench.csv")
    for kwargs, extras in _iter_flydsl_csv_cases():
        ret = test_gemm(**kwargs)
        ret.update(extras)
        written = append_tuned_op_bench_rows(
            bench_csv,
            [ret],
            op_name="gemm_a8w8",
        )
        if written:
            aiter.logger.info(
                "gemm_a8w8: appended %d tuned op bench row(s) to %s",
                written,
                bench_csv,
            )

if not args.no_legacy:
    if args.csv is not None:
        if not os.path.exists(args.csv):
            raise FileNotFoundError(f"CSV file not found: {args.csv}")
        shapes_df = pd.read_csv(args.csv)
        print(f"Loaded {len(shapes_df)} shapes from {args.csv}", flush=True)
        args.mnk = list(
            zip(
                shapes_df["M"].tolist(),
                shapes_df["N"].tolist(),
                shapes_df["K"].tolist(),
            )
        )

    df = test_normal_gemm_a8w8_pertoken_quant(
        args.dtype, args.quantDtype, args.mnk, args.pad_a
    )
    if get_gfx() != "gfx1250":
        test_skinny_gemm_a8w8_pertoken_quant()

    if args.output and df is not None:
        os.makedirs(args.output, exist_ok=True)
        if args.csv:
            csv_filename = os.path.basename(args.csv).replace(
                ".csv", f"_{args.suffix}.csv"
            )
        else:
            csv_filename = f"gemm_a8w8_{args.suffix}.csv"
        out_path = os.path.join(args.output, csv_filename)
        df.to_csv(out_path, index=False)
        print(f"Saved legacy results to: {out_path}")

    if args.bpreshuffle_csv is not None:
        if not os.path.exists(args.bpreshuffle_csv):
            raise FileNotFoundError(
                f"bpreshuffle CSV not found: {args.bpreshuffle_csv}"
            )
        bpre_df = pd.read_csv(args.bpreshuffle_csv)
        print(
            f"Loaded {len(bpre_df)} bpreshuffle shapes from {args.bpreshuffle_csv}",
            flush=True,
        )
        bpre_mnk = list(
            zip(
                bpre_df["M"].tolist(),
                bpre_df["N"].tolist(),
                bpre_df["K"].tolist(),
            )
        )
        df_bpre = test_normal_gemm_a8w8_pertoken_quant(
            args.dtype, args.quantDtype, bpre_mnk, args.pad_a, skip_ck=True
        )
        if args.output and df_bpre is not None:
            bpre_filename = os.path.basename(args.bpreshuffle_csv).replace(
                ".csv", f"_{args.suffix}.csv"
            )
            bpre_out = os.path.join(args.output, bpre_filename)
            df_bpre.to_csv(bpre_out, index=False)
            print(f"Saved bpreshuffle results to: {bpre_out}")
