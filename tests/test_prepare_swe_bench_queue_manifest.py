from pathlib import Path
import importlib.util
import json
import sys


def _load_queue_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "prepare_swe_bench_queue_manifest.py"
    spec = importlib.util.spec_from_file_location("prepare_swe_bench_queue_manifest", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _prediction_task_manifest(tmp_path: Path) -> dict[str, object]:
    return {
        "report_kind": "swe_bench_prediction_task_manifest",
        "tasks": [
            {
                "instance_id": "django__django-1",
                "model_name_or_path": "agentkernel",
                "repo": "django/django",
                "base_commit": "abc123",
                "repo_cache_root": str(tmp_path / "repos"),
                "patch_path": str(tmp_path / "patches" / "django__django-1.diff"),
                "problem_statement": "Fix timezone parsing regression. TypeError: got an unexpected keyword argument 'header_rows'",
                "hints_text": "Look at parser edge cases.",
                "candidate_files": ["django/utils/dateparse.py", "tests/test_time.py"],
                "source_context": [
                    {
                        "path": "django/utils/dateparse.py",
                        "content": "def parse_datetime(value):\n    return value\n",
                        "truncated": False,
                    }
                ],
                "fail_to_pass": ["tests/test_time.py::test_parse"],
                "pass_to_pass": ["tests/test_time.py::test_existing"],
            }
        ],
    }


def test_build_swe_queue_manifest_creates_external_tasks(tmp_path):
    module = _load_queue_module()

    manifest = module.build_swe_queue_manifest(_prediction_task_manifest(tmp_path))

    assert manifest["manifest_kind"] == "swe_bench_patch_generation_queue_manifest"
    assert len(manifest["tasks"]) == 1
    task = manifest["tasks"][0]
    assert task["task_id"] == "swe_patch_django__django-1"
    assert task["workspace_subdir"] == "swe_bench_predictions/swe_patch_django__django-1"
    assert task["expected_files"] == ["patch.diff"]
    assert task["setup_commands"] == []
    assert "patch.diff" in task["success_command"]
    assert "django/utils/dateparse\\.py" in task["success_command"]
    assert "placeholder" in task["success_command"]
    assert "django/utils/dateparse.py" in task["prompt"]
    assert "source_context/django/utils/dateparse.py" in task["prompt"]
    assert "source_lines/django/utils/dateparse.py.lines" in task["prompt"]
    assert "line-numbered files to choose exact hunk anchors" in task["prompt"]
    assert "swe_patch_builder --path <candidate-path>" in task["prompt"]
    assert "--replace-lines <start> <end>" in task["prompt"]
    assert "Do not use git show" in task["prompt"]
    assert "sed -n '1,120p' astropy/<path>" in task["prompt"]
    assert "do not run git" in task["prompt"]
    assert "fake imports" in task["prompt"]
    assert "applies cleanly" in task["prompt"]
    assert "must change executable Python behavior" in task["prompt"]
    assert "docstrings, module headers" in task["prompt"]
    assert "Do not delete or rename existing production function/class definitions" in task["prompt"]
    assert "do not invent files" in task["prompt"]
    assert "def parse_datetime(value):" in task["prompt"]
    assert "High-value executable edit windows:" in task["prompt"]
    assert "django/utils/dateparse.py::parse_datetime" in task["prompt"]
    assert "ranked by issue identifiers, hints, tests, and source tokens" in task["prompt"]
    assert "Required issue identifiers:" in task["prompt"]
    assert "header_rows" in task["prompt"]
    assert task["metadata"]["swe_instance_id"] == "django__django-1"
    assert task["metadata"]["swe_candidate_files"] == ["django/utils/dateparse.py", "tests/test_time.py"]
    assert task["metadata"]["swe_fail_to_pass"] == ["tests/test_time.py::test_parse"]
    assert task["metadata"]["swe_pass_to_pass"] == ["tests/test_time.py::test_existing"]
    assert "django/utils/dateparse.py::parse_datetime" in task["metadata"]["swe_executable_edit_windows"]
    assert task["metadata"]["setup_file_contents"] == {
        "django/utils/dateparse.py": "def parse_datetime(value):\n    return value\n",
        "source_context/django/utils/dateparse.py": "def parse_datetime(value):\n    return value\n",
        "source_lines/django/utils/dateparse.py.lines": "   1: def parse_datetime(value):\n   2:     return value\n   3: \n",
    }
    assert task["metadata"]["semantic_verifier"]["kind"] == "swe_patch_apply_check"
    assert task["metadata"]["semantic_verifier"]["repo_cache_root"] == str(tmp_path / "repos")
    assert task["metadata"]["semantic_verifier"]["expected_changed_paths"] == [
        "django/utils/dateparse.py",
        "tests/test_time.py",
    ]
    assert task["metadata"]["semantic_verifier"]["required_patch_identifiers"] == ["header_rows"]
    assert task["metadata"]["swe_patch_output_path"].endswith("django__django-1.diff")
    assert task["metadata"]["workflow_guard"]["managed_paths"] == [
        "patch.diff",
        "django/utils/dateparse.py",
        "source_context/django/utils/dateparse.py",
        "source_lines/django/utils/dateparse.py.lines",
    ]


def test_build_swe_queue_manifest_preserves_focused_duplicate_source_context(tmp_path):
    module = _load_queue_module()
    base = _prediction_task_manifest(tmp_path)
    task = base["tasks"][0]
    task["source_context"] = [
        {
            "path": "django/utils/dateparse.py",
            "content": "# large prefix\n",
            "truncated": True,
        },
        {
            "path": "django/utils/dateparse.py",
            "content": "def parse_datetime(value):\n    return value\n",
            "truncated": True,
            "context_kind": "focused_window",
            "focus_term": "parse_datetime",
            "line_start": 321,
            "line_end": 322,
        },
    ]

    manifest = module.build_swe_queue_manifest(base)
    generated = manifest["tasks"][0]

    assert generated["metadata"]["setup_file_contents"]["django/utils/dateparse.py"] == "# large prefix\n"
    assert "source_context/django/utils/dateparse.py.2_parse_datetime" in generated["metadata"]["setup_file_contents"]
    assert (
        generated["metadata"]["setup_file_contents"]["source_context/django/utils/dateparse.py.2_parse_datetime"]
        == "def parse_datetime(value):\n    return value\n"
    )
    assert generated["metadata"]["setup_file_contents"][
        "source_lines/django/utils/dateparse.py.lines.2_parse_datetime.lines"
    ] == " 321: def parse_datetime(value):\n 322:     return value\n 323: \n"
    assert "django/utils/dateparse.py::parse_datetime" in generated["metadata"]["swe_executable_edit_windows"]


def test_build_swe_queue_manifest_skips_tasks_without_source_context(tmp_path):
    module = _load_queue_module()
    manifest = module.build_swe_queue_manifest(
        {
            "tasks": [
                {
                    "instance_id": "scikit-learn__scikit-learn-25931",
                    "repo": "scikit-learn/scikit-learn",
                    "base_commit": "abc123",
                    "repo_cache_root": str(tmp_path / "repos"),
                    "patch_path": str(tmp_path / "patches" / "sklearn.diff"),
                    "problem_statement": "IsolationForest warns about valid feature names.",
                    "hints_text": "",
                    "candidate_files": ["sklearn/ensemble/_iforest.py"],
                    "source_context": [],
                    "fail_to_pass": ["test_iforest"],
                    "pass_to_pass": [],
                }
            ]
        }
    )

    assert manifest["tasks"] == []
    assert manifest["skipped_task_count"] == 1
    assert manifest["skipped_tasks"] == [
        {
            "instance_id": "scikit-learn__scikit-learn-25931",
            "repo": "scikit-learn/scikit-learn",
            "reason": "missing_source_context",
            "candidate_files": ["sklearn/ensemble/_iforest.py"],
            "repo_cache_root": str(tmp_path / "repos"),
        }
    ]


def test_prepare_swe_bench_queue_manifest_cli_writes_manifest(tmp_path, monkeypatch, capsys):
    module = _load_queue_module()
    input_path = tmp_path / "prediction_tasks.json"
    output_path = tmp_path / "queue_manifest.json"
    input_path.write_text(json.dumps(_prediction_task_manifest(tmp_path)), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_swe_bench_queue_manifest.py",
            "--prediction-task-manifest",
            str(input_path),
            "--output-manifest-json",
            str(output_path),
        ],
    )

    module.main()

    manifest = json.loads(output_path.read_text(encoding="utf-8"))
    assert len(manifest["tasks"]) == 1
    assert f"output_manifest_json={output_path}" in capsys.readouterr().out


def test_build_swe_queue_manifest_accepts_workspace_prefix(tmp_path):
    module = _load_queue_module()

    manifest = module.build_swe_queue_manifest(
        _prediction_task_manifest(tmp_path),
        workspace_prefix="swe_source_probe",
    )

    assert manifest["tasks"][0]["workspace_subdir"] == "swe_source_probe/swe_patch_django__django-1"


def test_build_swe_queue_manifest_ranks_issue_token_edit_windows(tmp_path):
    module = _load_queue_module()
    manifest = module.build_swe_queue_manifest(
        {
            "tasks": [
                {
                    "instance_id": "astropy__astropy-14365",
                    "repo": "astropy/astropy",
                    "base_commit": "abc123",
                    "repo_cache_root": str(tmp_path / "repos"),
                    "patch_path": str(tmp_path / "patches" / "astropy.diff"),
                    "problem_statement": "QDP commands should be case insensitive: read serr 1 2 should match.",
                    "hints_text": "The regex that searches for QDP commands is not case insensitive.",
                    "candidate_files": ["astropy/io/ascii/qdp.py"],
                    "source_context": [
                        {
                            "path": "astropy/io/ascii/qdp.py",
                            "content": (
                                "def _get_tables_from_qdp_file(qdp_file):\n"
                                "    err_specs = {}\n"
                                "    return err_specs\n\n"
                                "def _line_type(line):\n"
                                "    _command_re = r\"READ [TS]ERR(\\\\s+[0-9]+)+\"\n"
                                "    _line_type_re = re.compile(_command_re)\n"
                                "    return 'command' if _line_type_re.match(line) else 'data'\n"
                            ),
                            "truncated": False,
                        }
                    ],
                    "fail_to_pass": ["astropy/io/ascii/tests/test_qdp.py::test_roundtrip[True]"],
                    "pass_to_pass": [],
                }
            ]
        }
    )

    prompt = manifest["tasks"][0]["prompt"]
    window_section = prompt.split("High-value executable edit windows:", 1)[1]

    assert window_section.index("astropy/io/ascii/qdp.py::_line_type") < window_section.index(
        "astropy/io/ascii/qdp.py::_get_tables_from_qdp_file"
    )


def test_build_swe_queue_manifest_suggests_case_insensitive_regex_command(tmp_path):
    module = _load_queue_module()
    manifest = module.build_swe_queue_manifest(
        {
            "tasks": [
                {
                    "instance_id": "astropy__astropy-14365",
                    "repo": "astropy/astropy",
                    "base_commit": "abc123",
                    "repo_cache_root": str(tmp_path / "repos"),
                    "patch_path": str(tmp_path / "patches" / "astropy.diff"),
                    "problem_statement": "QDP commands should be case insensitive: read serr 1 2 should match.",
                    "hints_text": "The regex that searches for QDP commands is not case insensitive.",
                    "candidate_files": ["astropy/io/ascii/qdp.py"],
                    "source_context": [
                        {
                            "path": "astropy/io/ascii/qdp.py",
                            "content": (
                                "import re\n\n"
                                "def _line_type(line):\n"
                                "    _type_re = r\"READ SERR\"\n"
                                "    _line_type_re = re.compile(_type_re)\n"
                                "    return _line_type_re.match(line)\n\n"
                                "def _read_values(line):\n"
                                "    for v in line.split():\n"
                                "        if v == \"NO\":\n"
                                "            pass\n"
                            ),
                            "truncated": False,
                        }
                    ],
                    "fail_to_pass": [],
                    "pass_to_pass": [],
                }
            ]
        }
    )

    task = manifest["tasks"][0]

    assert "Suggested targeted patch commands:" in task["prompt"]
    assert "first patch-writing action should execute one of these commands exactly" in task["prompt"]
    assert "re.compile(_type_re, re.IGNORECASE)" in task["prompt"]
    assert task["metadata"]["swe_suggested_patch_commands"] == [
        "swe_patch_builder --path astropy/io/ascii/qdp.py "
        "--replace-line 5 --with '    _line_type_re = re.compile(_type_re, re.IGNORECASE)' "
        "--replace-line 10 --with '        if v.upper() == \"NO\":' > patch.diff",
    ]


def test_build_swe_queue_manifest_suggests_rst_header_rows_command(tmp_path):
    module = _load_queue_module()
    manifest = module.build_swe_queue_manifest(
        {
            "tasks": [
                {
                    "instance_id": "astropy__astropy-14182",
                    "repo": "astropy/astropy",
                    "base_commit": "abc123",
                    "repo_cache_root": str(tmp_path / "repos"),
                    "patch_path": str(tmp_path / "patches" / "astropy.diff"),
                    "problem_statement": "Please support header rows in RestructuredText output using header_rows.",
                    "hints_text": "",
                    "candidate_files": ["astropy/io/ascii/rst.py"],
                    "source_context": [
                        {
                            "path": "astropy/io/ascii/rst.py",
                            "content": (
                                "class RST(FixedWidth):\n"
                                "    def __init__(self):\n"
                                "        super().__init__(delimiter_pad=None, bookend=False)\n"
                                "\n"
                                "    def write(self, lines):\n"
                                "        lines = super().write(lines)\n"
                                "        lines = [lines[1]] + lines + [lines[1]]\n"
                                "        return lines\n"
                            ),
                            "truncated": False,
                        }
                    ],
                    "fail_to_pass": [],
                    "pass_to_pass": [],
                }
            ]
        }
    )

    task = manifest["tasks"][0]

    assert "Suggested targeted patch commands:" in task["prompt"]
    assert "header_rows=header_rows" in task["prompt"]
    assert "idx = len(self.header.header_rows)" in task["prompt"]
    assert task["metadata"]["swe_suggested_patch_commands"] == [
        "swe_patch_builder --path astropy/io/ascii/rst.py --replace-lines 2 8 "
        "--with '    def __init__(self, header_rows=None):' "
        "--with '        super().__init__(delimiter_pad=None, bookend=False, header_rows=header_rows)' "
        "--with '' "
        "--with '    def write(self, lines):' "
        "--with '        lines = super().write(lines)' "
        "--with '        idx = len(self.header.header_rows)' "
        "--with '        return [lines[idx]] + lines + [lines[idx]]' "
        "--with '' "
        "--with '    def read(self, table):' "
        "--with '        self.data.start_line = 2 + len(self.header.header_rows)' "
        "--with '        return super().read(table)' > patch.diff"
    ]


def test_build_swe_queue_manifest_suggests_nested_separability_command(tmp_path):
    module = _load_queue_module()
    manifest = module.build_swe_queue_manifest(
        {
            "tasks": [
                {
                    "instance_id": "astropy__astropy-12907",
                    "repo": "astropy/astropy",
                    "base_commit": "abc123",
                    "repo_cache_root": str(tmp_path / "repos"),
                    "patch_path": str(tmp_path / "patches" / "astropy.diff"),
                    "problem_statement": "separability_matrix is wrong for nested CompoundModels.",
                    "hints_text": "",
                    "candidate_files": ["astropy/modeling/separable.py"],
                    "source_context": [
                        {
                            "path": "astropy/modeling/separable.py",
                            "content": (
                                "def _cstack(left, right):\n"
                                "    if isinstance(right, Model):\n"
                                "        cright = _coord_matrix(right, 'right', noutp)\n"
                                "    else:\n"
                                "        cright = np.zeros((noutp, right.shape[1]))\n"
                                "        cright[-right.shape[0]:, -right.shape[1]:] = 1\n"
                                "    return np.hstack([cleft, cright])\n"
                            ),
                            "truncated": False,
                        }
                    ],
                    "fail_to_pass": [],
                    "pass_to_pass": [],
                }
            ]
        }
    )

    task = manifest["tasks"][0]

    assert "Suggested targeted patch commands:" in task["prompt"]
    assert "cright[-right.shape[0]:, -right.shape[1]:] = right" in task["prompt"]
    assert task["metadata"]["swe_suggested_patch_commands"] == [
        "swe_patch_builder --path astropy/modeling/separable.py --replace-line 6 --with '        cright[-right.shape[0]:, -right.shape[1]:] = right' > patch.diff"
    ]


def test_build_swe_queue_manifest_suggests_nddata_operand_mask_command(tmp_path):
    module = _load_queue_module()
    manifest = module.build_swe_queue_manifest(
        {
            "tasks": [
                {
                    "instance_id": "astropy__astropy-14995",
                    "repo": "astropy/astropy",
                    "base_commit": "abc123",
                    "repo_cache_root": str(tmp_path / "repos"),
                    "patch_path": str(tmp_path / "patches" / "astropy.diff"),
                    "problem_statement": (
                        "NDDataRef mask propagation fails when bitwise_or sees operand.mask is None."
                    ),
                    "hints_text": "The _arithmetic_mask branch should preserve self.mask.",
                    "candidate_files": ["astropy/nddata/mixins/ndarithmetic.py"],
                    "source_context": [
                        {
                            "path": "astropy/nddata/mixins/ndarithmetic.py",
                            "content": (
                                "    def _arithmetic_mask(self, operation, operand, handle_mask):\n"
                                "        if self.mask is None and operand is None:\n"
                                "            return None\n"
                                "        elif self.mask is None and operand is not None:\n"
                                "            return deepcopy(operand.mask)\n"
                                "        elif operand is None:\n"
                                "            return deepcopy(self.mask)\n"
                            ),
                            "truncated": False,
                        }
                    ],
                    "fail_to_pass": [],
                    "pass_to_pass": [],
                }
            ]
        }
    )

    task = manifest["tasks"][0]

    assert "Suggested targeted patch commands:" in task["prompt"]
    assert "operand.mask is None" in task["prompt"]
    assert task["metadata"]["swe_suggested_patch_commands"] == [
        "swe_patch_builder --path astropy/nddata/mixins/ndarithmetic.py --replace-line 6 --with '        elif operand is None or operand.mask is None:' > patch.diff"
    ]


def test_build_swe_queue_manifest_suggests_django_username_validator_commands(tmp_path):
    module = _load_queue_module()
    manifest = module.build_swe_queue_manifest(
        {
            "tasks": [
                {
                    "instance_id": "django__django-11099",
                    "repo": "django/django",
                    "base_commit": "abc123",
                    "repo_cache_root": str(tmp_path / "repos"),
                    "patch_path": str(tmp_path / "patches" / "django.diff"),
                    "problem_statement": "UsernameValidator allows trailing newline in usernames.",
                    "hints_text": "",
                    "candidate_files": ["django/contrib/auth/validators.py"],
                    "source_context": [
                        {
                            "path": "django/contrib/auth/validators.py",
                            "content": (
                                "class ASCIIUsernameValidator(validators.RegexValidator):\n"
                                "    regex = r'^[\\w.@+-]+$'\n"
                                "\n"
                                "class UnicodeUsernameValidator(validators.RegexValidator):\n"
                                "    regex = r'^[\\w.@+-]+$'\n"
                            ),
                            "truncated": False,
                        }
                    ],
                    "fail_to_pass": [],
                    "pass_to_pass": [],
                }
            ]
        }
    )

    task = manifest["tasks"][0]

    assert "Suggested targeted patch commands:" in task["prompt"]
    assert r"^[\w.@+-]+\Z" in task["prompt"]
    assert task["metadata"]["swe_suggested_patch_commands"] == [
        'swe_patch_builder --path django/contrib/auth/validators.py --replace-lines 2 5 '
        '--with "    regex = r\'^[\\w.@+-]+\\Z\'" '
        '--with "" '
        '--with "class UnicodeUsernameValidator(validators.RegexValidator):" '
        '--with "    regex = r\'^[\\w.@+-]+\\Z\'" > patch.diff',
    ]


def test_build_swe_queue_manifest_suggests_django_memoryview_command(tmp_path):
    module = _load_queue_module()
    manifest = module.build_swe_queue_manifest(
        {
            "tasks": [
                {
                    "instance_id": "django__django-11133",
                    "repo": "django/django",
                    "base_commit": "abc123",
                    "repo_cache_root": str(tmp_path / "repos"),
                    "patch_path": str(tmp_path / "patches" / "django.diff"),
                    "problem_statement": "HttpResponse doesn't handle memoryview objects",
                    "hints_text": "",
                    "candidate_files": ["django/http/response.py"],
                    "source_context": [
                        {
                            "path": "django/http/response.py",
                            "content": (
                                "    def make_bytes(self, value):\n"
                                "        if isinstance(value, bytes):\n"
                                "            return bytes(value)\n"
                            ),
                            "truncated": False,
                        }
                    ],
                    "fail_to_pass": [],
                    "pass_to_pass": [],
                }
            ]
        }
    )

    task = manifest["tasks"][0]

    assert "Suggested targeted patch commands:" in task["prompt"]
    assert "memoryview" in task["prompt"]
    assert task["metadata"]["swe_suggested_patch_commands"] == [
        "swe_patch_builder --path django/http/response.py --replace-line 2 --with '        if isinstance(value, (bytes, memoryview)):' > patch.diff"
    ]


def test_build_swe_queue_manifest_suggests_django_slugify_strip_command(tmp_path):
    module = _load_queue_module()
    manifest = module.build_swe_queue_manifest(
        {
            "tasks": [
                {
                    "instance_id": "django__django-12983",
                    "repo": "django/django",
                    "base_commit": "abc123",
                    "repo_cache_root": str(tmp_path / "repos"),
                    "patch_path": str(tmp_path / "patches" / "django.diff"),
                    "problem_statement": "Make django.utils.text.slugify() strip dashes and underscores.",
                    "hints_text": "",
                    "candidate_files": ["django/utils/text.py"],
                    "source_context": [
                        {
                            "path": "django/utils/text.py",
                            "content": "    return re.sub(r'[-\\s]+', '-', value)\n",
                            "truncated": False,
                        }
                    ],
                    "fail_to_pass": [],
                    "pass_to_pass": [],
                }
            ]
        }
    )

    task = manifest["tasks"][0]

    assert "Suggested targeted patch commands:" in task["prompt"]
    assert ".strip('-_')" in task["prompt"]
    assert task["metadata"]["swe_suggested_patch_commands"] == [
        'swe_patch_builder --path django/utils/text.py --replace-line 1 --with "    return re.sub(r\'[-\\s]+\', \'-\', value).strip(\'-_\')" > patch.diff'
    ]


def test_build_swe_queue_manifest_suggests_requests_redirect_carry_forward_command(tmp_path):
    module = _load_queue_module()
    manifest = module.build_swe_queue_manifest(
        {
            "tasks": [
                {
                    "instance_id": "psf__requests-1963",
                    "repo": "psf/requests",
                    "base_commit": "abc123",
                    "repo_cache_root": str(tmp_path / "repos"),
                    "patch_path": str(tmp_path / "patches" / "requests.diff"),
                    "problem_statement": (
                        "Session.resolve_redirects copies the original request for all subsequent requests."
                    ),
                    "hints_text": "",
                    "candidate_files": ["requests/sessions.py"],
                    "source_context": [
                        {
                            "path": "requests/sessions.py",
                            "content": (
                                "            resp = self.send(prepared_request)\n"
                                "            i += 1\n"
                                "            yield resp\n"
                            ),
                            "truncated": False,
                        }
                    ],
                    "fail_to_pass": [],
                    "pass_to_pass": [],
                }
            ]
        }
    )

    task = manifest["tasks"][0]

    assert "Suggested targeted patch commands:" in task["prompt"]
    assert "req = prepared_request" in task["prompt"]
    assert task["metadata"]["swe_suggested_patch_commands"] == [
        'swe_patch_builder --path requests/sessions.py --replace-lines 2 3 '
        '--with "            i += 1" '
        '--with "            req = prepared_request" '
        '--with "            yield resp" > patch.diff'
    ]


def test_build_swe_queue_manifest_suggests_requests_hooks_list_command(tmp_path):
    module = _load_queue_module()
    manifest = module.build_swe_queue_manifest(
        {
            "tasks": [
                {
                    "instance_id": "psf__requests-863",
                    "repo": "psf/requests",
                    "base_commit": "abc123",
                    "repo_cache_root": str(tmp_path / "repos"),
                    "patch_path": str(tmp_path / "patches" / "requests.diff"),
                    "problem_statement": "Allow lists in the dict values of the hooks argument.",
                    "hints_text": "",
                    "candidate_files": ["requests/models.py"],
                    "source_context": [
                        {
                            "path": "requests/models.py",
                            "content": (
                                "        for (k, v) in list(hooks.items()):\n"
                                "            self.register_hook(event=k, hook=v)\n"
                            ),
                            "truncated": False,
                        }
                    ],
                    "fail_to_pass": [],
                    "pass_to_pass": [],
                }
            ]
        }
    )

    task = manifest["tasks"][0]

    assert "Suggested targeted patch commands:" in task["prompt"]
    assert "for hook in v" in task["prompt"]
    assert task["metadata"]["swe_suggested_patch_commands"] == [
        'swe_patch_builder --path requests/models.py --replace-lines 1 2 '
        '--with "        for (k, v) in list(hooks.items()):" '
        '--with "            if isinstance(v, (list, tuple)):" '
        '--with "                for hook in v:" '
        '--with "                    self.register_hook(event=k, hook=hook)" '
        '--with "            else:" '
        '--with "                self.register_hook(event=k, hook=v)" > patch.diff'
    ]


def test_build_swe_queue_manifest_suggests_sympy_empty_array_command(tmp_path):
    module = _load_queue_module()
    manifest = module.build_swe_queue_manifest(
        {
            "tasks": [
                {
                    "instance_id": "sympy__sympy-23117",
                    "repo": "sympy/sympy",
                    "base_commit": "abc123",
                    "repo_cache_root": str(tmp_path / "repos"),
                    "patch_path": str(tmp_path / "patches" / "sympy.diff"),
                    "problem_statement": "sympy.Array([]) fails with not enough values to unpack.",
                    "hints_text": "",
                    "candidate_files": ["sympy/tensor/array/ndim_array.py"],
                    "source_context": [
                        {
                            "path": "sympy/tensor/array/ndim_array.py",
                            "content": (
                                "        def f(pointer):\n"
                                "            if not isinstance(pointer, Iterable):\n"
                                "                return [pointer], ()\n"
                                "\n"
                                "            result = []\n"
                                "            elems, shapes = zip(*[f(i) for i in pointer])\n"
                            ),
                            "truncated": False,
                        }
                    ],
                    "fail_to_pass": [],
                    "pass_to_pass": [],
                }
            ]
        }
    )

    task = manifest["tasks"][0]

    assert "Suggested targeted patch commands:" in task["prompt"]
    assert "return result, (0,)" in task["prompt"]
    assert task["metadata"]["swe_suggested_patch_commands"] == [
        'swe_patch_builder --path sympy/tensor/array/ndim_array.py --replace-lines 5 6 '
        '--with "            result = []" '
        '--with "            if not pointer:" '
        '--with "                return result, (0,)" '
        '--with "            elems, shapes = zip(*[f(i) for i in pointer])" > patch.diff'
    ]


def test_build_swe_queue_manifest_suggests_sympy_uniq_changed_size_command(tmp_path):
    module = _load_queue_module()
    manifest = module.build_swe_queue_manifest(
        {
            "tasks": [
                {
                    "instance_id": "sympy__sympy-18835",
                    "repo": "sympy/sympy",
                    "base_commit": "abc123",
                    "repo_cache_root": str(tmp_path / "repos"),
                    "patch_path": str(tmp_path / "patches" / "sympy.diff"),
                    "problem_statement": "uniq modifies list argument instead of raising changed size during iteration.",
                    "hints_text": "",
                    "candidate_files": ["sympy/utilities/iterables.py"],
                    "source_context": [
                        {
                            "path": "sympy/utilities/iterables.py",
                            "content": (
                                "def uniq(seq, result=None):\n"
                                "    try:\n"
                                "        seen = set()\n"
                                "        result = result or []\n"
                                "        for i, s in enumerate(seq):\n"
                                "            if not (s in seen or seen.add(s)):\n"
                                "                yield s\n"
                                "    except TypeError:\n"
                                "        if s not in result:\n"
                                "            yield s\n"
                                "            result.append(s)\n"
                            ),
                            "truncated": True,
                            "context_kind": "focused_window",
                            "focus_term": "uniq",
                            "line_start": 2088,
                            "line_end": 2095,
                        }
                    ],
                    "fail_to_pass": ["test_uniq"],
                    "pass_to_pass": [],
                }
            ]
        }
    )

    task = manifest["tasks"][0]

    assert "sequence changed size during iteration" in task["prompt"]
    assert task["metadata"]["swe_suggested_patch_commands"] == [
        'swe_patch_builder --path sympy/utilities/iterables.py --replace-lines 2089 2098 '
        '--with "    try:" '
        '--with "        size = len(seq)" '
        '--with "    except TypeError:" '
        '--with "        size = None" '
        '--with "    try:" '
        '--with "        seen = set()" '
        '--with "        result = result or []" '
        '--with "        for i, s in enumerate(seq):" '
        '--with "            if size is not None and len(seq) != size:" '
        '--with "                raise RuntimeError(\'sequence changed size during iteration\')" '
        '--with "            if not (s in seen or seen.add(s)):" '
        '--with "                yield s" '
        '--with "                if size is not None and len(seq) != size:" '
        '--with "                    raise RuntimeError(\'sequence changed size during iteration\')" '
        '--with "    except TypeError:" '
        '--with "        if s not in result:" '
        '--with "            yield s" '
        '--with "            if size is not None and len(seq) != size:" '
        '--with "                raise RuntimeError(\'sequence changed size during iteration\')" '
        '--with "            result.append(s)" > patch.diff'
    ]


def test_build_swe_queue_manifest_suggests_sympy_partitions_copy_command(tmp_path):
    module = _load_queue_module()
    manifest = module.build_swe_queue_manifest(
        {
            "tasks": [
                {
                    "instance_id": "sympy__sympy-20154",
                    "repo": "sympy/sympy",
                    "base_commit": "abc123",
                    "repo_cache_root": str(tmp_path / "repos"),
                    "patch_path": str(tmp_path / "patches" / "sympy.diff"),
                    "problem_statement": "partitions() reusing the output dictionaries should copy dictionaries.",
                    "hints_text": "",
                    "candidate_files": ["sympy/utilities/iterables.py"],
                    "source_context": [
                        {
                            "path": "sympy/utilities/iterables.py",
                            "content": (
                                "    if size:\n"
                                "        yield sum(ms.values()), ms\n"
                                "    else:\n"
                                "        yield ms\n"
                                "\n"
                                "    while keys != [1]:\n"
                                "        room -= need\n"
                                "        if size:\n"
                                "            yield sum(ms.values()), ms\n"
                                "        else:\n"
                                "            yield ms\n"
                            ),
                            "truncated": True,
                            "context_kind": "focused_window",
                            "focus_term": "partitions",
                            "line_start": 1804,
                            "line_end": 1814,
                        }
                    ],
                    "fail_to_pass": ["test_partitions"],
                    "pass_to_pass": [],
                }
            ]
        }
    )

    task = manifest["tasks"][0]

    assert "Suggested targeted patch commands:" in task["prompt"]
    assert "ms.copy()" in task["prompt"]
    assert task["metadata"]["swe_suggested_patch_commands"] == [
        'swe_patch_builder --path sympy/utilities/iterables.py --replace-lines 1805 1814 '
        '--with "        yield sum(ms.values()), ms.copy()" '
        '--with "    else:" '
        '--with "        yield ms.copy()" '
        '--with "" '
        '--with "    while keys != [1]:" '
        '--with "        room -= need" '
        '--with "        if size:" '
        '--with "            yield sum(ms.values()), ms.copy()" '
        '--with "        else:" '
        '--with "            yield ms.copy()" > patch.diff'
    ]


def test_build_swe_queue_manifest_suggests_sympy_col_insert_index_command(tmp_path):
    module = _load_queue_module()
    manifest = module.build_swe_queue_manifest(
        {
            "tasks": [
                {
                    "instance_id": "sympy__sympy-13647",
                    "repo": "sympy/sympy",
                    "base_commit": "abc123",
                    "repo_cache_root": str(tmp_path / "repos"),
                    "patch_path": str(tmp_path / "patches" / "sympy.diff"),
                    "problem_statement": "Matrix.col_insert() no longer seems to work correctly.",
                    "hints_text": "",
                    "candidate_files": ["sympy/matrices/common.py"],
                    "source_context": [
                        {
                            "path": "sympy/matrices/common.py",
                            "content": (
                                "    def _eval_col_insert(self, pos, other):\n"
                                "        cols = self.cols\n"
                                "\n"
                                "        def entry(i, j):\n"
                                "            if j < pos:\n"
                                "                return self[i, j]\n"
                                "            elif pos <= j < pos + other.cols:\n"
                                "                return other[i, j - pos]\n"
                                "            return self[i, j - pos - other.cols]\n"
                            ),
                            "truncated": True,
                            "context_kind": "focused_window",
                            "focus_term": "col_insert",
                            "line_start": 81,
                            "line_end": 89,
                        }
                    ],
                    "fail_to_pass": ["test_col_insert"],
                    "pass_to_pass": [],
                }
            ]
        }
    )

    task = manifest["tasks"][0]

    assert "Suggested targeted patch commands:" in task["prompt"]
    assert task["metadata"]["swe_suggested_patch_commands"] == [
        'swe_patch_builder --path sympy/matrices/common.py --replace-line 89 '
        "--with '            return self[i, j - other.cols]' > patch.diff"
    ]


def test_build_swe_queue_manifest_suggests_sympy_morse_one_mapping_command(tmp_path):
    module = _load_queue_module()
    manifest = module.build_swe_queue_manifest(
        {
            "tasks": [
                {
                    "instance_id": "sympy__sympy-16886",
                    "repo": "sympy/sympy",
                    "base_commit": "abc123",
                    "repo_cache_root": str(tmp_path / "repos"),
                    "patch_path": str(tmp_path / "patches" / "sympy.diff"),
                    "problem_statement": (
                        'Morse encoding for "1" is not correct. '
                        'The correct mapping is ".----": "1".'
                    ),
                    "hints_text": "",
                    "candidate_files": ["sympy/crypto/crypto.py"],
                    "source_context": [
                        {
                            "path": "sympy/crypto/crypto.py",
                            "content": (
                                "morse_char = {\n"
                                '    "-----": "0", "----": "1",\n'
                                '    "..---": "2",\n'
                                "}\n"
                            ),
                            "truncated": True,
                            "context_kind": "focused_window",
                            "focus_term": "morse",
                            "line_start": 76,
                            "line_end": 79,
                        }
                    ],
                    "fail_to_pass": ["test_encode_morse"],
                    "pass_to_pass": [],
                }
            ]
        }
    )

    task = manifest["tasks"][0]

    assert task["metadata"]["swe_suggested_patch_commands"] == [
        'swe_patch_builder --path sympy/crypto/crypto.py --replace-line 77 '
        '--with \'    "-----": "0", ".----": "1",\' > patch.diff'
    ]
    assert task["suggested_commands"] == task["metadata"]["swe_suggested_patch_commands"]
    setup_crypto = task["metadata"]["setup_file_contents"]["sympy/crypto/crypto.py"].splitlines()
    assert setup_crypto[76] == '    "-----": "0", "----": "1",'
    assert "source_lines/sympy/crypto/crypto.py.lines" in task["metadata"]["workflow_guard"]["managed_paths"]
    assert "sympy/crypto/crypto.py" in task["metadata"]["workflow_guard"]["managed_paths"]


def test_build_swe_queue_manifest_suggests_sympy_vector_radd_zero_command(tmp_path):
    module = _load_queue_module()
    manifest = module.build_swe_queue_manifest(
        {
            "tasks": [
                {
                    "instance_id": "sympy__sympy-14711",
                    "repo": "sympy/sympy",
                    "base_commit": "abc123",
                    "repo_cache_root": str(tmp_path / "repos"),
                    "patch_path": str(tmp_path / "patches" / "sympy.diff"),
                    "problem_statement": "vector add 0 error with sum([N.x, (0 * N.x)])",
                    "hints_text": "",
                    "candidate_files": ["sympy/physics/vector/vector.py"],
                    "source_context": [
                        {
                            "path": "sympy/physics/vector/vector.py",
                            "content": (
                                "    def __add__(self, other):\n"
                                '        """The add operator for Vector. """\n'
                                "        other = _check_vector(other)\n"
                                "        return Vector(self.args + other.args)\n"
                                "\n"
                                "    def __xor__(self, other):\n"
                                "        return Vector(outlist)\n"
                                "\n"
                                "    _sympystr = __str__\n"
                                "    __radd__ = __add__\n"
                                "    __rand__ = __and__\n"
                            ),
                            "truncated": True,
                            "context_kind": "focused_window",
                            "focus_term": "radd",
                            "line_start": 56,
                            "line_end": 65,
                        }
                    ],
                    "fail_to_pass": ["test_vector_add_zero"],
                    "pass_to_pass": [],
                }
            ]
        }
    )

    task = manifest["tasks"][0]

    assert task["metadata"]["swe_suggested_patch_commands"] == [
        "swe_patch_builder --path sympy/physics/vector/vector.py "
        "--replace-lines 58 58 "
        "--with '        if other == 0:' "
        "--with '            return self' "
        "--with '        other = _check_vector(other)' "
        "--replace-lines 65 65 "
        "--with '    def __radd__(self, other):' "
        "--with '        return self.__add__(other)' > patch.diff"
    ]


def test_build_swe_queue_manifest_suggests_sympy_intersection_deduplicate_command(tmp_path):
    module = _load_queue_module()
    manifest = module.build_swe_queue_manifest(
        {
            "tasks": [
                {
                    "instance_id": "sympy__sympy-16988",
                    "repo": "sympy/sympy",
                    "base_commit": "abc123",
                    "repo_cache_root": str(tmp_path / "repos"),
                    "patch_path": str(tmp_path / "patches" / "sympy.diff"),
                    "problem_statement": "Intersection should remove duplicates.",
                    "hints_text": "",
                    "candidate_files": ["sympy/sets/sets.py"],
                    "source_context": [
                        {
                            "path": "sympy/sets/sets.py",
                            "content": (
                                "class Union(Set, LatticeOp):\n"
                                "    def __new__(cls, *args, **kwargs):\n"
                                "        if True:\n"
                                "            args = list(cls._new_args_filter(args))\n"
                                "            return simplify_union(args)\n"
                                "\n"
                                "class Intersection(Set, LatticeOp):\n"
                                "    def __new__(cls, *args, **kwargs):\n"
                                "        evaluate = kwargs.get('evaluate', global_evaluate[0])\n"
                                "        args = _sympify(args)\n"
                                "        if evaluate:\n"
                                "            args = list(cls._new_args_filter(args))\n"
                                "            return simplify_intersection(args)\n"
                            ),
                            "truncated": True,
                            "context_kind": "focused_window",
                            "focus_term": "Intersection",
                            "line_start": 1259,
                            "line_end": 1268,
                        }
                    ],
                    "fail_to_pass": ["test_intersection"],
                    "pass_to_pass": [],
                }
            ]
        }
    )

    task = manifest["tasks"][0]

    assert "Suggested targeted patch commands:" in task["prompt"]
    assert task["metadata"]["swe_suggested_patch_commands"] == [
        'swe_patch_builder --path sympy/sets/sets.py --replace-line 1270 '
        '--with "            args = list(ordered(set(cls._new_args_filter(args)), Set._infimum_key))" > patch.diff'
    ]


def test_build_swe_queue_manifest_suggests_sympy_matmul_matrix_only_commands(tmp_path):
    module = _load_queue_module()
    manifest = module.build_swe_queue_manifest(
        {
            "tasks": [
                {
                    "instance_id": "sympy__sympy-13773",
                    "repo": "sympy/sympy",
                    "base_commit": "abc123",
                    "repo_cache_root": str(tmp_path / "repos"),
                    "patch_path": str(tmp_path / "patches" / "sympy.diff"),
                    "problem_statement": "__matmul__ should return NotImplemented for scalar operands.",
                    "hints_text": "Matrix @ non-matrix scalar should not behave like scalar multiplication.",
                    "candidate_files": ["sympy/matrices/common.py"],
                    "source_context": [
                        {
                            "path": "sympy/matrices/common.py",
                            "content": (
                                "    @call_highest_priority('__rmatmul__')\n"
                                "    def __matmul__(self, other):\n"
                                "        return self.__mul__(other)\n"
                                "\n"
                                "    @call_highest_priority('__matmul__')\n"
                                "    def __rmatmul__(self, other):\n"
                                "        return self.__rmul__(other)\n"
                            ),
                            "truncated": True,
                            "context_kind": "focused_window",
                            "focus_term": "__matmul__",
                            "line_start": 1973,
                            "line_end": 1979,
                        }
                    ],
                    "fail_to_pass": ["test_matmul_scalar"],
                    "pass_to_pass": [],
                }
            ]
        }
    )

    task = manifest["tasks"][0]

    assert "Suggested targeted patch commands:" in task["prompt"]
    commands = task["metadata"]["swe_suggested_patch_commands"]
    assert len(commands) == 1
    assert commands[0].startswith("swe_patch_builder --path sympy/matrices/common.py --replace-line 1975 ")
    assert (
        '--replace-line 1975 --with "        return self.__mul__(other) if getattr(_matrixify(other), '
        "'is_Matrix', False) or getattr(_matrixify(other), 'is_MatrixLike', False) else NotImplemented\""
    ) in commands[0]
    assert (
        '--replace-line 1979 --with "        return self.__rmul__(other) if getattr(_matrixify(other), '
        "'is_Matrix', False) or getattr(_matrixify(other), 'is_MatrixLike', False) else NotImplemented\""
    ) in commands[0]
    assert commands[0].endswith(" > patch.diff")


def test_build_swe_queue_manifest_suggests_sympy_tall_upper_bounds_command(tmp_path):
    module = _load_queue_module()
    manifest = module.build_swe_queue_manifest(
        {
            "tasks": [
                {
                    "instance_id": "sympy__sympy-12454",
                    "repo": "sympy/sympy",
                    "base_commit": "abc123",
                    "repo_cache_root": str(tmp_path / "repos"),
                    "patch_path": str(tmp_path / "patches" / "sympy.diff"),
                    "problem_statement": "is_upper() raises IndexError for tall matrices.",
                    "hints_text": "test_hessenberg also covers tall upper hessenberg.",
                    "candidate_files": ["sympy/matrices/matrices.py"],
                    "source_context": [
                        {
                            "path": "sympy/matrices/matrices.py",
                            "content": (
                                "    def _eval_is_upper_hessenberg(self):\n"
                                "        return all(self[i, j].is_zero\n"
                                "                   for i in range(2, self.rows)\n"
                                "                   for j in range(i - 1))\n"
                                "\n"
                                "    @property\n"
                                "    def is_upper(self):\n"
                                "        return all(self[i, j].is_zero\n"
                                "                   for i in range(1, self.rows)\n"
                                "                   for j in range(i))\n"
                            ),
                            "truncated": True,
                            "context_kind": "focused_window",
                            "focus_term": "is_upper",
                            "line_start": 621,
                            "line_end": 630,
                        }
                    ],
                    "fail_to_pass": ["test_is_upper", "test_hessenberg"],
                    "pass_to_pass": [],
                }
            ]
        }
    )

    task = manifest["tasks"][0]
    commands = task["metadata"]["swe_suggested_patch_commands"]

    assert len(commands) == 1
    assert commands[0] == (
        'swe_patch_builder --path sympy/matrices/matrices.py '
        '--replace-line 624 --with "                   for j in range(min(i - 1, self.cols)))" '
        '--replace-line 630 --with "                   for j in range(min(i, self.cols)))" > patch.diff'
    )


def test_build_swe_queue_manifest_suggests_sklearn_iforest_fit_score_command(tmp_path):
    module = _load_queue_module()
    manifest = module.build_swe_queue_manifest(
        {
            "tasks": [
                {
                    "instance_id": "scikit-learn__scikit-learn-25931",
                    "repo": "scikit-learn/scikit-learn",
                    "base_commit": "abc123",
                    "repo_cache_root": str(tmp_path / "repos"),
                    "patch_path": str(tmp_path / "patches" / "sklearn.diff"),
                    "problem_statement": (
                        "IsolationForest warns that X does not have valid feature names "
                        "when contamination is not auto."
                    ),
                    "hints_text": "fit calls score_samples and revalidates the already validated input.",
                    "candidate_files": ["sklearn/ensemble/_iforest.py"],
                    "source_context": [
                        {
                            "path": "sklearn/ensemble/_iforest.py",
                            "content": (
                                "    def fit(self, X, y=None, sample_weight=None):\n"
                                "        if self.contamination == \"auto\":\n"
                                "            self.offset_ = -0.5\n"
                                "            return self\n"
                                "\n"
                                "        self.offset_ = np.percentile(self.score_samples(X), 100.0 * self.contamination)\n"
                                "\n"
                                "        return self\n"
                            ),
                            "truncated": True,
                            "context_kind": "focused_window",
                            "focus_term": "contamination",
                            "line_start": 343,
                            "line_end": 350,
                        }
                    ],
                    "fail_to_pass": ["test_iforest"],
                    "pass_to_pass": [],
                }
            ]
        }
    )

    commands = manifest["tasks"][0]["metadata"]["swe_suggested_patch_commands"]

    assert commands == [
        'swe_patch_builder --path sklearn/ensemble/_iforest.py --replace-line 348 '
        "--with '        self.offset_ = np.percentile(-self._compute_chunked_score_samples(X), "
        "100.0 * self.contamination)' > patch.diff"
    ]


def test_build_swe_queue_manifest_suggests_sklearn_show_versions_joblib_command(tmp_path):
    module = _load_queue_module()
    manifest = module.build_swe_queue_manifest(
        {
            "tasks": [
                {
                    "instance_id": "scikit-learn__scikit-learn-14141",
                    "repo": "scikit-learn/scikit-learn",
                    "base_commit": "abc123",
                    "repo_cache_root": str(tmp_path / "repos"),
                    "patch_path": str(tmp_path / "patches" / "sklearn.diff"),
                    "problem_statement": "Add joblib in show_versions dependencies.",
                    "hints_text": "joblib should be listed by _get_deps_info.",
                    "candidate_files": ["sklearn/utils/_show_versions.py"],
                    "source_context": [
                        {
                            "path": "sklearn/utils/_show_versions.py",
                            "content": (
                                "    deps = [\n"
                                "        \"pip\",\n"
                                "        \"setuptools\",\n"
                                "        \"sklearn\",\n"
                                "        \"numpy\",\n"
                                "        \"scipy\",\n"
                                "        \"Cython\",\n"
                                "        \"pandas\",\n"
                                "    ]\n"
                            ),
                            "truncated": True,
                            "context_kind": "focused_window",
                            "focus_term": "joblib",
                            "line_start": 29,
                            "line_end": 38,
                        }
                    ],
                    "fail_to_pass": ["test_get_deps_info"],
                    "pass_to_pass": [],
                }
            ]
        }
    )

    commands = manifest["tasks"][0]["metadata"]["swe_suggested_patch_commands"]

    assert commands == [
        "swe_patch_builder --path sklearn/utils/_show_versions.py --replace-lines 35 35 "
        "--with '        \"Cython\",' --with '        \"joblib\",' > patch.diff"
    ]


def test_build_swe_queue_manifest_suggests_pylint_sys_path_guard_command(tmp_path):
    module = _load_queue_module()
    manifest = module.build_swe_queue_manifest(
        {
            "tasks": [
                {
                    "instance_id": "pylint-dev__pylint-7277",
                    "repo": "pylint-dev/pylint",
                    "base_commit": "abc123",
                    "repo_cache_root": str(tmp_path / "repos"),
                    "patch_path": str(tmp_path / "patches" / "pylint.diff"),
                    "problem_statement": (
                        "pylint removes first item from sys.path when running from runpy. "
                        "It should check the first item before removing."
                    ),
                    "hints_text": "runpy can put a non-working-directory entry first in sys.path.",
                    "candidate_files": ["pylint/__init__.py"],
                    "source_context": [
                        {
                            "path": "pylint/__init__.py",
                            "content": (
                                "def modify_sys_path() -> None:\n"
                                "    \"\"\"Modify sys path for execution as Python module.\"\"\"\n"
                                "    sys.path.pop(0)\n"
                                "    env_pythonpath = os.environ.get(\"PYTHONPATH\", \"\")\n"
                            ),
                            "truncated": True,
                            "context_kind": "focused_window",
                            "focus_term": "modify_sys_path",
                            "line_start": 80,
                            "line_end": 103,
                        }
                    ],
                    "fail_to_pass": ["test_modify_sys_path"],
                    "pass_to_pass": [],
                }
            ]
        }
    )

    commands = manifest["tasks"][0]["metadata"]["swe_suggested_patch_commands"]

    assert commands == [
        'swe_patch_builder --path pylint/__init__.py --replace-lines 82 82 '
        '--with \'    if sys.path[0] in ("", ".", os.getcwd()):\' '
        "--with '        sys.path.pop(0)' > patch.diff"
    ]


def test_build_swe_queue_manifest_suggests_pytest_caplog_clear_in_place_command(tmp_path):
    module = _load_queue_module()
    manifest = module.build_swe_queue_manifest(
        {
            "tasks": [
                {
                    "instance_id": "pytest-dev__pytest-10051",
                    "repo": "pytest-dev/pytest",
                    "base_commit": "abc123",
                    "repo_cache_root": str(tmp_path / "repos"),
                    "patch_path": str(tmp_path / "patches" / "pytest.diff"),
                    "problem_statement": (
                        "caplog.get_records and caplog.clear conflict: after caplog.clear() "
                        "get_records is frozen and decoupled from actual caplog records."
                    ),
                    "hints_text": "clear replaces rather than clears the shared records list.",
                    "candidate_files": ["src/_pytest/logging.py"],
                    "source_context": [
                        {
                            "path": "src/_pytest/logging.py",
                            "content": (
                                "class LogCaptureFixture:\n"
                                "\n"
                                "    def clear(self) -> None:\n"
                                "        self.handler.reset()\n"
                                "\n"
                                "    def set_level(self, level):\n"
                                "        pass\n"
                            ),
                            "truncated": True,
                            "context_kind": "focused_window",
                            "focus_term": "clear",
                            "line_start": 357,
                            "line_end": 445,
                        }
                    ],
                    "fail_to_pass": ["test_caplog_clear"],
                    "pass_to_pass": [],
                }
            ]
        }
    )

    commands = manifest["tasks"][0]["metadata"]["swe_suggested_patch_commands"]

    assert commands == [
        'swe_patch_builder --path src/_pytest/logging.py --replace-lines 360 360 '
        '--with "        self.handler.records.clear()" '
        '--with "        self.handler.stream = StringIO()" > patch.diff'
    ]


def test_build_swe_queue_manifest_suggests_sphinx_version_comparison_command(tmp_path):
    module = _load_queue_module()
    manifest = module.build_swe_queue_manifest(
        {
            "tasks": [
                {
                    "instance_id": "sphinx-doc__sphinx-9711",
                    "repo": "sphinx-doc/sphinx",
                    "base_commit": "abc123",
                    "repo_cache_root": str(tmp_path / "repos"),
                    "patch_path": str(tmp_path / "patches" / "sphinx.diff"),
                    "problem_statement": (
                        "needs_extensions checks versions using strings, so version 0.10 "
                        "is treated as older than 0.6."
                    ),
                    "hints_text": "",
                    "candidate_files": ["sphinx/extension.py"],
                    "source_context": [
                        {
                            "path": "sphinx/extension.py",
                            "content": (
                                "from typing import TYPE_CHECKING, Any, Dict\n"
                                "\n"
                                "from sphinx.config import Config\n"
                                "\n"
                                "def verify_needs_extensions(app, config):\n"
                                "    for extname, reqversion in config.needs_extensions.items():\n"
                                "        extension = app.extensions.get(extname)\n"
                                "        if extension.version == 'unknown version' or reqversion > extension.version:\n"
                                "            raise VersionRequirementError(__('This project needs the extension %s at least in '\n"
                                "                                             'version %s and therefore cannot be built with '\n"
                                "                                             'the loaded version (%s).') %\n"
                                "                                          (extname, reqversion, extension.version))\n"
                            ),
                            "truncated": True,
                            "context_kind": "focused_window",
                            "focus_term": "needs_extensions",
                            "line_start": 11,
                            "line_end": 58,
                        }
                    ],
                    "fail_to_pass": ["test_needs_extensions"],
                    "pass_to_pass": [],
                }
            ]
        }
    )

    command = manifest["tasks"][0]["metadata"]["swe_suggested_patch_commands"][0]

    assert "from packaging.version import InvalidVersion, Version" in command
    assert "Version(reqversion) > Version(extension.version)" in command
    assert "except InvalidVersion" in command


def test_build_swe_queue_manifest_suggests_matplotlib_stackplot_local_color_cycle(tmp_path):
    module = _load_queue_module()
    manifest = module.build_swe_queue_manifest(
        {
            "tasks": [
                {
                    "instance_id": "matplotlib__matplotlib-24026",
                    "repo": "matplotlib/matplotlib",
                    "base_commit": "abc123",
                    "repo_cache_root": str(tmp_path / "repos"),
                    "patch_path": str(tmp_path / "patches" / "matplotlib.diff"),
                    "problem_statement": (
                        "stackplot should not change Axes cycler and should accept C2 "
                        "cycle reference colors instead of raising a ValueError."
                    ),
                    "hints_text": "",
                    "candidate_files": ["lib/matplotlib/stackplot.py"],
                    "source_context": [
                        {
                            "path": "lib/matplotlib/stackplot.py",
                            "content": (
                                "import numpy as np\n"
                                "\n"
                                "def stackplot(axes, x, *args, colors=None):\n"
                                "    y = np.row_stack(args)\n"
                                "    if colors is not None:\n"
                                "        axes.set_prop_cycle(color=colors)\n"
                                "    color = axes._get_lines.get_next_color()\n"
                                "    axes.fill_between(x, y[0], y[1], facecolor=color)\n"
                                "    for i in range(len(y) - 1):\n"
                                "        color = axes._get_lines.get_next_color()\n"
                            ),
                            "truncated": True,
                            "context_kind": "focused_window",
                            "focus_term": "stackplot",
                            "line_start": 9,
                            "line_end": 120,
                        }
                    ],
                    "fail_to_pass": ["test_stackplot_colors"],
                    "pass_to_pass": [],
                }
            ]
        }
    )

    command = manifest["tasks"][0]["metadata"]["swe_suggested_patch_commands"][0]

    assert "import itertools" in command
    assert "colors = itertools.cycle(colors)" in command
    assert "colors = (axes._get_lines.get_next_color() for _ in y)" in command
    assert '--replace-line 15 --with "    color = next(colors)"' in command
    assert '--replace-line 18 --with "        color = next(colors)"' in command


def test_build_swe_queue_manifest_suggests_xarray_chunks_no_load_command(tmp_path):
    module = _load_queue_module()
    manifest = module.build_swe_queue_manifest(
        {
            "tasks": [
                {
                    "instance_id": "pydata__xarray-6721",
                    "repo": "pydata/xarray",
                    "base_commit": "abc123",
                    "repo_cache_root": str(tmp_path / "repos"),
                    "patch_path": str(tmp_path / "patches" / "xarray.diff"),
                    "problem_statement": (
                        "Accessing chunks on zarr backed xarray seems to load entire array "
                        "into memory instead of staying lazy."
                    ),
                    "hints_text": "Accessing .chunks should not trigger eager loading of the data.",
                    "candidate_files": ["xarray/core/common.py"],
                    "source_context": [
                        {
                            "path": "xarray/core/common.py",
                            "content": (
                                "def get_chunksizes(\n"
                                "    variables,\n"
                                "):\n"
                                "    chunks = {}\n"
                                "    for v in variables:\n"
                                "        if hasattr(v.data, \"chunks\"):\n"
                                "            for dim, c in v.chunksizes.items():\n"
                                "                pass\n"
                            ),
                            "truncated": True,
                            "context_kind": "focused_window",
                            "focus_term": "chunks",
                            "line_start": 2020,
                            "line_end": 2032,
                        }
                    ],
                    "fail_to_pass": ["test_chunks_does_not_load_data"],
                    "pass_to_pass": [],
                }
            ]
        }
    )

    commands = manifest["tasks"][0]["metadata"]["swe_suggested_patch_commands"]

    assert commands == [
        'swe_patch_builder --path xarray/core/common.py --replace-line 2025 '
        '--with \'        if hasattr(v._data, "chunks"):\' > patch.diff'
    ]


def test_build_swe_queue_manifest_suggests_pytest_setuponly_saferepr_command(tmp_path):
    module = _load_queue_module()
    manifest = module.build_swe_queue_manifest(
        {
            "tasks": [
                {
                    "instance_id": "pytest-dev__pytest-7205",
                    "repo": "pytest-dev/pytest",
                    "base_commit": "abc123",
                    "repo_cache_root": str(tmp_path / "repos"),
                    "patch_path": str(tmp_path / "patches" / "pytest.diff"),
                    "problem_statement": (
                        "BytesWarning when using --setup-show with bytes parameter. "
                        "Should use saferepr instead of implicit str()."
                    ),
                    "hints_text": "Use saferepr with a shorter maxsize.",
                    "candidate_files": ["src/_pytest/setuponly.py"],
                    "source_context": [
                        {
                            "path": "src/_pytest/setuponly.py",
                            "content": (
                                "import pytest\n"
                                "\n"
                                "def _show_fixture_action(fixturedef, msg):\n"
                                "    if hasattr(fixturedef, \"cached_param\"):\n"
                                "        tw.write(\"[{}]\".format(fixturedef.cached_param))\n"
                            ),
                            "truncated": True,
                            "context_kind": "focused_window",
                            "focus_term": "saferepr",
                            "line_start": 1,
                            "line_end": 69,
                        }
                    ],
                    "fail_to_pass": ["test_setup_show_bytes"],
                    "pass_to_pass": [],
                }
            ]
        }
    )

    command = manifest["tasks"][0]["metadata"]["swe_suggested_patch_commands"][0]

    assert "--replace-lines 5 5" in command
    assert "from _pytest._io.saferepr import saferepr" in command
    assert "saferepr(fixturedef.cached_param, maxsize=42)" in command


def test_build_swe_queue_manifest_suggests_astropy_timeseries_required_columns_command(tmp_path):
    module = _load_queue_module()
    manifest = module.build_swe_queue_manifest(
        {
            "tasks": [
                {
                    "instance_id": "astropy__astropy-13033",
                    "repo": "astropy/astropy",
                    "base_commit": "abc123",
                    "repo_cache_root": str(tmp_path / "repos"),
                    "patch_path": str(tmp_path / "patches" / "astropy.diff"),
                    "problem_statement": (
                        "TimeSeries: misleading exception when required column check fails. "
                        "ValueError expected 'time' as the first columns but found 'time'."
                    ),
                    "hints_text": "The relevant code assumes time is the only required column.",
                    "candidate_files": ["astropy/timeseries/core.py"],
                    "source_context": [
                        {
                            "path": "astropy/timeseries/core.py",
                            "content": (
                                "    def _check_required_columns(self):\n"
                                "        required_columns = self._required_columns\n"
                                "        plural = 's' if len(required_columns) > 1 else ''\n"
                                "        if False:\n"
                                "            pass\n"
                                "        elif self.colnames[:len(required_columns)] != required_columns:\n"
                                "            raise ValueError(\"{} object is invalid - expected '{}' \"\n"
                                "                             \"as the first column{} but found '{}'\"\n"
                                "                             .format(self.__class__.__name__, required_columns[0], plural, self.colnames[0]))\n"
                            ),
                            "truncated": True,
                            "context_kind": "focused_window",
                            "focus_term": "_check_required_columns",
                            "line_start": 57,
                            "line_end": 81,
                        }
                    ],
                    "fail_to_pass": ["astropy/timeseries/tests/test_sampled.py::test_required_columns"],
                    "pass_to_pass": [],
                }
            ]
        }
    )

    command = manifest["tasks"][0]["metadata"]["swe_suggested_patch_commands"][0]

    assert "--replace-lines 63 65" in command
    assert "if len(required_columns) == 1:" in command
    assert "expected" in command
    assert "required_columns[0], plural," in command
    assert "required_columns, plural," in command
    assert "self.colnames[:len(required_columns)]" in command
    assert "Suggested targeted patch commands:" in manifest["tasks"][0]["prompt"]


def test_build_swe_queue_manifest_emits_windows_for_truncated_python(tmp_path):
    module = _load_queue_module()
    manifest = module.build_swe_queue_manifest(
        {
            "tasks": [
                {
                    "instance_id": "astropy__astropy-14365",
                    "repo": "astropy/astropy",
                    "base_commit": "abc123",
                    "repo_cache_root": str(tmp_path / "repos"),
                    "patch_path": str(tmp_path / "patches" / "astropy.diff"),
                    "problem_statement": "The QDP command regex is not case insensitive for read serr.",
                    "hints_text": "The regex that searches for QDP commands is not case insensitive.",
                    "candidate_files": ["astropy/io/ascii/qdp.py"],
                    "source_context": [
                        {
                            "path": "astropy/io/ascii/qdp.py",
                            "content": (
                                "def _line_type(line):\n"
                                "    _command_re = r\"READ [TS]ERR(\\\\s+[0-9]+)+\"\n"
                                "    _line_type_re = re.compile(_command_re)\n"
                                "    return _line_type_re.match(line)\n\n"
                                "class QDP:\n"
                                "    def read(self, table):\n"
                                "        return _read_table_qdp(\n"
                            ),
                            "truncated": True,
                        }
                    ],
                    "fail_to_pass": [],
                    "pass_to_pass": [],
                }
            ]
        }
    )

    prompt = manifest["tasks"][0]["prompt"]

    assert "High-value executable edit windows:" in prompt
    assert "astropy/io/ascii/qdp.py::_line_type" in prompt


def test_candidate_file_success_command_rejects_unrelated_diff():
    module = _load_queue_module()

    command = module._candidate_file_success_command(["django/utils/dateparse.py"])

    assert "grep -Eq 'django/utils/dateparse\\.py' patch.diff" in command
    assert "! grep -Eiq" not in command
