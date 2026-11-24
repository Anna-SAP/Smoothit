// Two-language UI strings. Chinese is the primary audience; English is the fallback elsewhere.
import type { HintId } from '../engine/director';

type Dict = Record<string, string>;

const zh: Dict = {
  loading: '正在布置影棚…',
  loadingModels: '加载人像模型',
  loadingCamera: '打开相机',
  errorTitle: '无法启动',
  errorDenied: '需要相机权限才能拍照。请在系统设置中允许 Smoothit 使用相机，然后重试。',
  errorInsecure: '相机只能在 HTTPS 或 App 内使用。',
  errorUnavailable: '没有找到可用的相机。',
  errorWebgl: '此设备不支持所需的图形能力（WebGL2）。',
  retry: '重试',
  save: '保存',
  saved: '已保存',
  share: '分享',
  compare: '按住对比原图',
  close: '关闭',
  delete: '删除',
  gallery: '作品',
  galleryEmpty: '还没有作品，拍一张试试。',
  softAuto: '柔光 自动',
  softOn: '柔光 开',
  softOff: '柔光 关',
  developing: '显影中…',
  'hint.noFace': '把脸对准镜头',
  'hint.tooClose': '手机稍微拿远一点，五官比例更好',
  'hint.backlit': '背光较强，转向光源会更通透',
  'hint.lowAngle': '手机再举高一点',
  'hint.lowLight': '光线偏暗，已用屏幕柔光补光',
};

const en: Dict = {
  loading: 'Setting up the studio…',
  loadingModels: 'Loading portrait models',
  loadingCamera: 'Opening camera',
  errorTitle: "Can't start",
  errorDenied: 'Smoothit needs camera access. Allow it in system settings, then retry.',
  errorInsecure: 'The camera only works over HTTPS or inside the app.',
  errorUnavailable: 'No usable camera was found.',
  errorWebgl: 'This device lacks the required graphics support (WebGL2).',
  retry: 'Retry',
  save: 'Save',
  saved: 'Saved',
  share: 'Share',
  compare: 'Hold to compare',
  close: 'Close',
  delete: 'Delete',
  gallery: 'Shots',
  galleryEmpty: 'No shots yet. Take one.',
  softAuto: 'Soft light auto',
  softOn: 'Soft light on',
  softOff: 'Soft light off',
  developing: 'Developing…',
  'hint.noFace': 'Bring your face into the frame',
  'hint.tooClose': 'Hold the phone a little farther away',
  'hint.backlit': 'Strong backlight: turn toward the light',
  'hint.lowAngle': 'Raise the phone slightly',
  'hint.lowLight': 'Low light: screen soft light is on',
};

export const lang: 'zh' | 'en' = navigator.language?.toLowerCase().startsWith('zh') ? 'zh' : 'en';
const dict = lang === 'zh' ? zh : en;

export function t(key: string): string {
  return dict[key] ?? key;
}

export function hintText(id: HintId): string {
  return t(`hint.${id}`);
}
