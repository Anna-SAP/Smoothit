// Getting a finished picture out of the app: the photo library on native builds, the share sheet
// or a download on the web.

function isNative(): boolean {
  const cap = (window as unknown as { Capacitor?: { isNativePlatform?: () => boolean } }).Capacitor;
  return !!cap?.isNativePlatform?.();
}

function blobToBase64(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result).split(',')[1] ?? '');
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(blob);
  });
}

async function nativeShare(blob: Blob, filename: string): Promise<void> {
  const [{ Filesystem, Directory }, { Share }] = await Promise.all([
    import('@capacitor/filesystem'),
    import('@capacitor/share'),
  ]);
  const file = await Filesystem.writeFile({ path: filename, data: await blobToBase64(blob), directory: Directory.Cache });
  // The system sheet offers "Save Image" (to Photos) alongside every share target.
  await Share.share({ files: [file.uri] });
}

function download(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 10_000);
}

async function webShare(blob: Blob, filename: string): Promise<boolean> {
  const file = new File([blob], filename, { type: blob.type });
  if (!navigator.canShare?.({ files: [file] })) return false;
  try {
    await navigator.share({ files: [file] });
  } catch (err) {
    if ((err as DOMException)?.name !== 'AbortError') return false;
  }
  return true;
}

/** Share through the system sheet where one exists; otherwise fall back to a download. */
export async function sharePhoto(blob: Blob, filename: string): Promise<void> {
  if (isNative()) return nativeShare(blob, filename);
  if (!(await webShare(blob, filename))) download(blob, filename);
}

/** Save to the device. iOS browsers can only reach the photo library through the share sheet. */
export async function savePhoto(blob: Blob, filename: string): Promise<void> {
  if (isNative()) return nativeShare(blob, filename);
  const iOS = /iPad|iPhone|iPod/.test(navigator.userAgent) || (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);
  if (iOS && (await webShare(blob, filename))) return;
  download(blob, filename);
}

export function photoFilename(date = new Date()): string {
  const p = (n: number) => String(n).padStart(2, '0');
  return `Smoothit_${date.getFullYear()}${p(date.getMonth() + 1)}${p(date.getDate())}_${p(date.getHours())}${p(date.getMinutes())}${p(date.getSeconds())}.jpg`;
}
