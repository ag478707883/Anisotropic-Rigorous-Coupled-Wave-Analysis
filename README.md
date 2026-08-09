# ASYRCWA

ASYRCWA 是一个面向周期光学结构与各向异性分层介质的高性能 RCWA
（Rigorous Coupled-Wave Analysis，严格耦合波分析）求解器。项目以 C++20
实现数值核心，通过 pybind11 提供 Python 接口，适合光栅、超表面、光子晶体、
磁光非互易结构和热辐射器件的光谱、衍射、散射参数与场分布计算。

## 项目背景

RCWA 将周期介质及电磁场展开到 Fourier 谐波空间，并把 Maxwell 方程转化为
矩阵本征问题。它对规则周期结构非常高效，但有限阶截断下的材料不连续、TM
收敛、各向异性张量耦合和深层结构传播稳定性都需要专门处理。

本项目围绕这些问题实现了：

- 均匀多层介质的 Berreman 4x4/TMM 求解；
- 一维层状光栅和二维周期图案的 RCWA 求解；
- 标量、各向异性和磁光介电常数/磁导率张量；
- Li Fourier 分解：界面法向分量使用逆规则，切向分量使用直接规则；
- 一维 TE/TM 正确分块，以及二维坐标对齐结构的有序 `L_y L_x` 分解；
- 稳定散射矩阵级联，避免厚层、金属和倏逝模导致的传递矩阵数值失稳；
- RETICOLO 矩形谐波阶数、`li=1` 有序 Li 分解和稳定散射矩阵级联；
- 反射、透射、吸收、衍射级次、S 参数、零级复振幅、方向热辐射通道和场平面输出；
- C++ 静态库与 Python `asyrcwa` 接口。

项目采用时间约定 `exp(-i*omega*t + i*k0*k.r)`，长度和波长接口默认使用
微米。Li 分解的公式、适用范围和文献对应关系见
[docs/li_fourier_factorization.md](docs/li_fourier_factorization.md)。Python 与 C++
的完整签名、参数约束、返回字段和高级诊断接口见
[docs/rcwa_api_reference.md](docs/rcwa_api_reference.md)。

### 代码结构

```text
include/rcwa/   C++ 公共接口与内部算法声明
src/rcwa/       C++ 数值核心
bindings/       pybind11 绑定
asyrcwa/        Python 包与运行时加载逻辑
examples/       论文结构复现和完整计算脚本
materials/      材料数据
tests/          C++ 回归与物理基准测试
docs/           算法推导和技术说明
```

## 启动方式

### 环境要求

当前仓库主要在 Windows + MinGW-w64 环境验证。需要：

- CMake 3.20 或更高版本；
- 支持 C++20 的编译器，预设使用 MinGW-w64；
- Python 3.9 或更高版本；
- Intel oneAPI MKL，必须包含 LAPACKE、CBLAS 和 `mkl_rt`；
- FFTW3。未检测到本地 FFTW3 时，CMake 会下载并静态构建 FFTW 3.3.11；
- 首次配置时需要网络，以便在缺少本地依赖时获取 FFTW3 和 Catch2。

如果 MKL 不在默认 oneAPI 目录中，请设置 `MKLROOT`：

```powershell
$env:MKLROOT = "D:\path\to\oneAPI\mkl\latest"
```

### 推荐：构建 C++ 与 Python 后端

在 PowerShell 中执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .

cmake --preset release
cmake --build --preset release --parallel
ctest --preset release
```

构建产物位于 `build/`，Python 扩展位于 `build/python/`。验证 Python 包是否找到
最新后端：

```powershell
python -m asyrcwa
```

`asyrcwa` 会自动注册 oneAPI、MinGW 和扩展目录的 Windows DLL 搜索路径。如果使用
自定义构建目录，可显式指定：

```powershell
$env:RCWA_BUILD_DIR = (Resolve-Path .\build-release).Path
python -m asyrcwa
```

### 仅构建 C++ 核心

```powershell
cmake -S . -B build-cpp `
  -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DRCWA_ENABLE_PYTHON_BINDINGS=OFF
cmake --build build-cpp --parallel
ctest --test-dir build-cpp --output-on-failure
```

下游 CMake 项目可通过 `add_subdirectory` 引入本仓库，并链接静态库目标 `rcwa`。

### 常用构建选项

| 选项 | 默认值 | 说明 |
| --- | --- | --- |
| `RCWA_ENABLE_PYTHON_BINDINGS` | `ON` | 构建 `rcwa_cpp` Python 扩展 |
| `RCWA_SINGLE_PRECISION` | `OFF` | 使用单精度实数/复数和 FFTW3f |
| `RCWA_ENABLE_IPO` | `ON` | Release 构建启用可用的 IPO/LTO |

## 实例

### 最小一维光栅光谱

下面的代码计算空气中的硅/空气二元光栅在 0.9-1.1 um 范围内的 TE/TM
反射、透射和吸收：

```python
import numpy as np
import asyrcwa

simulation = asyrcwa.New(
    Lattice=1.0,
    Orders=(3, 0),
)

simulation.SetMaterial("Air", 1.0 + 0.0j)
simulation.SetMaterial("Si", 12.0 + 0.0j)
simulation.SetSuperstrate("Air")
simulation.SetSubstrate("Air")
simulation.AddLayer("grating", 0.35, "Air")
simulation.SetRegionRectangle(
    "grating",
    "Si",
    Center=(0.0, 0.0),
    Angle=0.0,
    Halfwidths=(0.25, 0.5),
)
simulation.SetExcitationPlanewave((0.0, 0.0))

wavelengths_um = np.linspace(0.9, 1.1, 101)
spectrum = simulation.GetSpectrum(
    wavelengths_um,
    ("TE", "TM"),
    Workers=1,
)

for row in spectrum[::25]:
    print(
        row["lambda_um"],
        row["pol"],
        row["R"],
        row["T"],
        row["A"],
        row["conservation"],
    )
```

提高 `Orders` 会增加 Fourier 谐波数量和计算成本。正式计算应对谐波阶数、图案
采样分辨率和可选 Fourier 公式进行联合收敛检查。

### 运行仓库实例

```powershell
# 均匀磁光多层结构
python examples\run_chen_2024_defect_inas_multilayer_fig2.py

# 一维嵌入式 InAs 光栅
python examples\run_zou_2025_embedded_inas_grating_fig3.py

# 二维方孔超表面
python examples\run_huang_2026_wsm_si_square_holes_fig2.py

# 高对比度光栅圆偏振器
python examples\run_mutlu_2012_hcg_circular_polarizer_spectrum.py
```

实例通常会将图像写入 `images/`，部分脚本也会生成数值数据。论文复现脚本包含较密
的波长扫描和较高谐波阶数，运行时间可能明显高于上面的最小示例。

## 未来方向

- 建立覆盖更多公开 RCWA/FEM/FDTD 数据的自动化交叉验证集；
- 完善曲线和斜边界的 normal-vector/Jones/Kottke 收敛模型及适用性诊断；
- 增加 GPU 或批量线性代数后端，优化大谐波二维结构和多波长扫描；
- 支持分布式参数扫描、断点续算和标准化结果缓存；
- 扩充材料数据库、温度依赖模型及实验数据拟合接口；
- 增加伴随梯度、拓扑优化和参数反演工作流；
- 完善 C++/Python API 文档、跨平台 CI、预编译 wheel 和版本化发布流程；
- 继续降低 Fourier 矩阵、本征模和层缓存的峰值内存占用。

## 验证与许可

提交代码前建议运行：

```powershell
cmake --build --preset release --parallel
ctest --preset release
```

当前 Release 回归包含 Li TE/TM 有效介质极限、有序二维分解、各向异性模式、能量
守恒和独立 S4 基准。项目使用 [MIT License](LICENSE)。
