# external/

第三方源码目录。除非明确要求,不修改这里的算法实现。

## vjepa2/

`external/vjepa2/` 是从上游 V-JEPA 2 代码库中抽取的**最小可运行子集**,用于提供
冻结的视觉编码器与 latent predictor。

来源:

```text
仓库:    git@github.com:YinqiBai962/ABDUCTIVE-WORLD.git (工作仓库 vjepa2-baiz)
路径:    app/, src/
commit:  d8a34f2e309904f8437fb29d2d34bc48c25eecca (2026-07-14, "对齐")
说明:    上游 app/ 与 src/ 在工作仓库中无本地修改
```

### 与上游的唯一差异:导入前缀

上游把 `app/` 与 `src/` 作为顶层命名空间包。本项目自己的代码也占用 `src/`
(见 `docs/manual/requirement.md` §2.1),两者会在 `import src.*` 上冲突,
因此 vendored 副本整体作为一个包导入,并改写导入前缀:

```text
from src.<x>  ->  from external.vjepa2.src.<x>
import src.<x> ->  import external.vjepa2.src.<x>
from app.<x>  ->  from external.vjepa2.app.<x>
import app.<x> ->  import external.vjepa2.app.<x>
```

改写只作用于行首的 import 语句,未改动任何算法逻辑。核对方式:

```bash
grep -rnE "^\s*(from|import)\s+(src|app)\." external/   # 应无输出
```

### 包含的模块(静态导入闭包)

```text
app/vjepa/{transforms,utils}.py
src/datasets/utils/video/{functional,randaugment,randerase,transforms}.py
src/masks/utils.py
src/models/{predictor,vision_transformer,attentive_pooler}.py
src/models/utils/{modules,patch_embed,pos_embs}.py
src/utils/{checkpoint_loader,logging,schedulers,tensors,wrappers}.py
```

未包含上游的 `app/vjepa_2_1/`、`app/vjepa_droid/`、`src/datasets/*` 其余部分、
`src/hub/` 等本项目未引用的模块。

### 与上游同步

上游更新后,重新抽取同一批文件并套用上面的前缀改写即可;不要为了同步而扩大
vendor 范围,除非新增了实际引用。
