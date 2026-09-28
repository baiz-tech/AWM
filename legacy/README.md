# legacy/

被取代的历史实验,按原样保留,**未做任何导入改写或 launcher 规范化**。

## 内容

| 目录 | 说明 |
|---|---|
| `vjepa2_naive_probe_v5_decoder_physionpp/` | Physion++ dynamics decoder 的早期迭代,已被 `src/experiments/vjepa2_physionpp_dynamics_decoder_seed239/`(源自 `_physionpp3`)取代 |
| `vjepa2_naive_probe_v5_decoder_physionpp2/` | 同上,`_physionpp3` 的前一版 |
| `vjepa2_naive_probe_v6_decoder_probeTest1_clevrerAll/` | CLEVRER probe 对照实验(probeTest 1) |
| `vjepa2_naive_probe_v6_decoder_probeTest2_clevrerAll/` | CLEVRER probe 对照实验(probeTest 2) |
| `vjepa2_naive_probe_v6_decoder_probeTest3_clevrerAll/` | CLEVRER probe 对照实验(probeTest 3) |
| `vjepa2_naive_probe_v6_decoder_probeTest4_clevrerAll/` | CLEVRER probe 对照实验(probeTest 4) |
| `tests/test_physionpp_probe_physionpp2.py` | 针对 `_physionpp2` 的测试,随之归档,依赖上面的 legacy 代码 |
| `tests/test_script_structure_vjepa2_naive.py` | 校验旧 Python launcher(`resolve_run` / `build_launch_command`)与旧 `run_all.sh` 脚本结构的测试;该 launcher 已被 `tools/launcher/launch_from_config.sh` 取代 |

## 重要限制

- 这里的代码**保留原来的导入路径**(`recipe.*`、`recipe_formal.*`、`app.*`、
  `src.*` 指向上游仓库),因此**在本仓库内不可直接运行、不可导入**。
- 没有任何 config 迁移到 `configs/`;原 config 仍在各自的 `configs/` 子目录下。
- 需要重新启用其中某个实验时,应按 `docs/migration/CONVENTIONS.md` 的规则完整迁移,
  而不是从 `legacy/` 直接 import。

保留它们只是为了避免历史实现丢失,并使 paper 中可能引用的旧版本仍可追溯。
