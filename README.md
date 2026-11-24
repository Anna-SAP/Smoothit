# Smoothit

一键拍出影楼级自拍的手机相机。不是“先拍、再修”——取景时就在测光、布光、构图、修肤，按下快门得到的就是成片。

- **零配置**：一个快门、一个翻转、四种风格（影楼 / 红毯 / 杂志 / 胶片）。曝光、白平衡、补光、构图、透视校正、修肤强度全部自动。
- **所见即所得**：取景器和成片走同一条 GPU 管线。
- **全部在设备上完成**：人脸网格和分割模型随 App 打包，照片不上传。
- **iOS 与 Android 共用一套影像引擎**（WebGL2 + 端侧模型），用 Capacitor 打包成原生 App，也可作为 PWA 直接用浏览器打开。

研究结论与方案依据见 [docs/RESEARCH.md](docs/RESEARCH.md)，实现细节见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

## 它在快门那一刻做了什么

| 影楼里的角色 | Smoothit 里的对应 |
| --- | --- |
| 选 85 mm 镜头、站远一点 | 虹膜测距 + 3D 网格重投影，校正近距离广角的“大鼻子” |
| 灯光师 | 从脸上解出实际光照，按影楼布光补光；暗光时屏幕变柔光箱并闪光 |
| 测光表、灰卡 | 对皮肤测光（眼白当灰卡，不漂白深肤色），以肤色为准的白平衡 |
| 摄影师构图、选片 | 自动裁切到 3:4、眼线在上三分；连拍 5 帧挑睁眼且不糊的一帧 |
| 修图师 | 分频修肤（留毛孔、去瑕疵、匀肤色）、亮眼、白牙、去眼下暗沉 |
| 调色师 | 高光滚降、影调曲线、分离色调、柔光、颗粒 |

## 运行

需要 Node 20+。

```bash
npm install
npm run dev          # http://localhost:5173 （首次会下载约 20 MB 模型到 public/models）
```

浏览器只对安全来源开放摄像头。电脑上 `localhost` 可以直接用；要在同一局域网的手机上打开开发服务器：

```bash
npm run dev:https    # 自签名证书，手机访问 https://<电脑IP>:5173 并信任证书
```

加 `?debug=1` 可以看到帧率和引擎的各项决策。

## 打包成原生 App

```bash
npm run build && npx cap sync

npx cap open ios        # 需要 macOS + Xcode：选择签名团队后运行到真机
npx cap open android    # 需要 Android Studio；或直接：
cd android && ./gradlew assembleDebug   # 产物：android/app/build/outputs/apk/debug/app-debug.apk
```

相机权限说明已写入 `Info.plist`（`NSCameraUsageDescription`）和 `AndroidManifest.xml`（`CAMERA`）。

## 测试

```bash
npm test             # 单元测试
npm run typecheck
```

视觉回归与端到端测试（无头 Chromium + 假摄像头）见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#测试)。

## 当前状态与已知边界

已在无头浏览器中用真实人像和合成的业余拍摄条件（暗光、逆光、偏色、过曝）端到端验证；Android 调试包可编译。
**尚未在真机上实测**：帧率、发热、各机型前摄视场角、WebView 相机权限流程都需要真机跑一轮。

软件改变不了物理：强逆光和硬顶光投下的阴影只能减轻；离镜头太近、机位太低时会给一句提示；
暗光成片的上限取决于传感器（原生多帧合成在路线图里）。详见研究文档第 4、5 节。

## 第三方组件

- [MediaPipe Tasks Vision](https://ai.google.dev/edge/mediapipe/solutions/guide)（Apache-2.0）：Face Landmarker、Selfie Multiclass Segmenter 模型与 WASM 运行时
- [Capacitor](https://capacitorjs.com/)（MIT）
