            if result.returncode != 0:
                print(f"\n=== {name} ({mp_label}) STDOUT ===\n{result.stdout[-2000:]}")
                print(f"\n=== {name} ({mp_label}) STDERR ===\n{result.stderr[-2000:]}")
            self.assertEqual(
                result.returncode,
                0,
                f"{name} ({mp_label}) tuner exited with code {result.returncode}",
            )
            self.assertTrue(
                os.path.exists(tuned), f"{name} ({mp_label}): tuned CSV not created"
            )

            df = pd.read_csv(tuned)
            df.columns = df.columns.str.strip()
            self.assertGreaterEqual(
                len(df),
                len(shapes),
                f"{name} ({mp_label}): expected >= {len(shapes)} rows",
            )
            for key in cfg["keys"]:
                self.assertIn(
                    key, df.columns, f"{name} ({mp_label}): missing column {key}"
                )
            for _, row in df.iterrows():
                us = float(row.get("us", -1))
                self.assertNotEqual(
                    us, 0, f"{name} ({mp_label}): us == 0 for {dict(row)}"
                )

    def test_a8w8_mp1(self):
        self._run_one("a8w8", mp=1)

    def test_a8w8_mp_default(self):
        self._run_one("a8w8", mp=None)

    def test_a8w8_blockscale_mp1(self):
        self._run_one("a8w8_blockscale", mp=1)

    def test_a8w8_blockscale_mp_default(self):
        self._run_one("a8w8_blockscale", mp=None)

    def test_a8w8_blockscale_bpreshuffle_asm_mp1(self):
        """Smoke-test the a8w8 blockscale B-preshuffle ASM tuner path."""
        cfg = self.TUNERS["a8w8_blockscale"]
        with tempfile.TemporaryDirectory() as tmp:
            untuned = os.path.join(tmp, "untuned.csv")
            tuned = os.path.join(tmp, "tuned.csv")
            _write_csv(untuned, cfg["header"], [(16, 1536, 7168)])

            result = _run_tuner(
                cfg["script"],
                untuned,
                tuned,
                extra_args=["--libtype", "asm", "--preshuffle", "--batch", "1"],
                timeout=900,
                mp=1,
            )
            if result.returncode != 0:
                print(
                    f"\n=== a8w8_blockscale bpreshuffle asm STDOUT ===\n{result.stdout[-2000:]}"
                )
                print(
                    f"\n=== a8w8_blockscale bpreshuffle asm STDERR ===\n{result.stderr[-2000:]}"
                )
            self.assertEqual(
                result.returncode,
                0,
                "a8w8_blockscale bpreshuffle asm tuner failed",
            )
            self.assertTrue(
                os.path.exists(tuned),
                "a8w8_blockscale bpreshuffle asm: tuned CSV not created",
            )

            df = pd.read_csv(tuned)
            df.columns = df.columns.str.strip()
            self.assertGreaterEqual(len(df), 1)
            self.assertTrue((df["libtype"] == "asm").any())
            self.assertTrue((df["errRatio"].astype(float) <= 0.05).all())

    def test_a8w8_bpreshuffle_mp1(self):
        self._run_one("a8w8_bpreshuffle", mp=1)

    def test_a8w8_bpreshuffle_mp_default(self):
        self._run_one("a8w8_bpreshuffle", mp=None)

    def test_batched_a8w8_mp1(self):
        self._run_one("batched_a8w8", mp=1)

    def test_batched_a8w8_mp_default(self):
        self._run_one("batched_a8w8", mp=None)

    def test_batched_bf16_mp1(self):
        self._run_one("batched_bf16", mp=1)

    def test_batched_bf16_mp_default(self):
        self._run_one("batched_bf16", mp=None)

    def test_fmoe_mp1(self):
        self._run_one("fmoe", mp=1)

