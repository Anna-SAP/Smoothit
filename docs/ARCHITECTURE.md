# 架构

## 一帧的旅程

预览和成片走**同一条管线**（`Engine.render`），区别只有工作分辨率和输出目标，所以取景器里看到的就是保存下来的。

```
摄像头帧 ──┬─► FaceLandmarker ─► 478 点 3D 网格 + 表情系数 ─► FaceGeometry（像素坐标、法线、姿态、测距）
           ├─► ImageSegmenter ─► 人物 / 头发 / 皮肤 / 衣服 掩膜（256 px，每 3 帧一次）
           └─► Engine.ingest ─► 镜像 + 工作分辨率 + 金字塔(1/2, 1/4, 1/8) ─► 192×256 回读
                                                                              │
                         analyze()：肤色、眼白、球谐光照、场景直方图、背景亮度 ◄─┘
                                                                              │
                 Director.update()：曝光 / 白平衡 / 补光增益 / 裁切 / 透视变形 / 提示
                                                                              │
Engine.render():                                                              ▼
  源空间   网格 → G-buffer（补光增益、眼/唇/口/皮肤分区）
          分割 → 引导滤波抠像
          皮肤掩膜 → 掩膜双边基底 → 区域肤色
          背景圆盘散景、辉光
          合成：分频修肤 + 五官 + 重打光
  输出空间 变形网格：裁切构图 + 镜头透视校正 → 白平衡/曝光/肩部/影调/分离色调/暗角/颗粒 → 屏幕或 JPEG
```

## 目录

| 路径 | 职责 |
| --- | --- |
| `src/vision/` | 端侧模型封装（`vision.ts`）和人脸网格拓扑：三角面、分区、取样点（`topology.ts`） |
| `src/engine/faceModel.ts` | 由关键点重建几何：法线、瞳距、虹膜测距、姿态、面部正面深度 |
| `src/engine/analysis.ts` | CPU 侧场景分析（“测光表和摄影师的眼睛”） |
| `src/engine/lighting.ts` | 球谐光照拟合、光照描述、是否需要纠正、逐顶点补光增益 |
| `src/engine/geometry.ts` | 推拉变焦式透视校正 → 平滑反向变形场 |
| `src/engine/composition.ts` | 自动构图与平滑跟随 |
| `src/engine/director.ts` | “导演”：把测量变成渲染参数；白平衡与曝光决策；提示 |
| `src/engine/looks.ts` | 四种风格（影楼 / 红毯 / 杂志 / 胶片）：只描述意图，不含场景适配 |
| `src/engine/engine.ts`, `shaders.ts`, `gl.ts` | WebGL2 管线与 GLSL ES 3.00 着色器 |
| `src/engine/bestshot.ts` | 连拍选片打分 |
| `src/capture/still.ts` | 单帧全质量成片（App 与 Lab 共用） |
| `src/camera/camera.ts` | 相机流、分辨率协商、曝光补偿（渐进增强） |
| `src/app/` | 界面、取景循环、拍摄流程、作品集（IndexedDB）、保存/分享 |
| `lab.html`, `src/lab.ts` | 开发用：静态图走成片管线，支持中间结果调试视图 |
| `android/`, `ios/` | Capacitor 原生壳 |

## 约定

- **纹理坐标**：所有离屏纹理 uv (0,0) 为图像**左上角**，因此关键点坐标可直接当纹理坐标用；只有画到屏幕的最后一步翻转 Y。
- **法线空间**：x 向右、y 向**上**、z 朝向镜头。MediaPipe 的 z 是“相对深度，越小越近”，尺度与 x 相同。
- **镜像**：前摄画面在 `ingest` 第一步镜像，关键点同步镜像，之后所有模块都只见到镜像后的空间。
- **色彩**：中间纹理为 8 位伽马编码；需要物理正确的运算（曝光、重打光、散景、辉光）在着色器内转线性光。
- **尺度**：所有模糊半径以瞳距为单位，脸大脸小处理一致。
- **时间**：`Director` 对曝光、白平衡、光照系数做指数平滑；主光侧向带迟滞，不会逐帧翻转。

## 调试

`lab.html?delegate=CPU` 可加载任意图片；`window.smoothitLab.run(url, look, { debug })` 的 `debug` 可取
`skin`（皮肤掩膜）、`light`（补光增益/覆盖/眼下）、`regions`（分区）、`base`（双边基底）、
`matte`（抠像）、`wide`（区域肤色）、`texture`（纹理频段）。

App 加 `?debug=1` 显示帧率、单帧耗时、质量档位和导演的各项决策。

## 性能设计

- 预览工作分辨率 1024（长边），三档自适应降级（1024 / 832 / 640，分割间隔 3 / 4 / 6 帧）。
- 重计算都在 1/4、1/8 分辨率完成：双边基底、抠像引导滤波、散景、辉光；全分辨率只有一次合成和一次输出。
- 场景分析每 2 帧一次，回读仅 192×256。
- 成片工作分辨率上限 3072（内存 ≤ 4 GB 的设备 2048）。

## 测试

```bash
npm test                    # 30 项单元测试（合成针孔相机下的通用人脸）
npm run dev                 # 另开终端
node scripts/visual-test.mjs --out out --look studio,film a.jpg b.jpg
node scripts/e2e.mjs --video face.y4m --out out
```

`e2e.mjs` 需要一段人脸视频作为假摄像头，例如：

```bash
ffmpeg -loop 1 -i portrait.jpg -vf "scale=960:1280:force_original_aspect_ratio=increase,crop=960:1280" -t 1.2 -r 10 -pix_fmt yuv420p face.y4m
```
