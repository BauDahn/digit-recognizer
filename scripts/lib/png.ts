/**
 * Minimal PNG decoder (8-bit, non-interlaced, gray / RGB / RGBA / gray+alpha) so the dataset
 * scripts do not need an image library. Returns the grayscale brightness (0..255) of every pixel.
 */
import { inflateSync } from 'node:zlib';

export interface GrayImage {
  width: number;
  height: number;
  gray: Uint8Array;
}

const CHANNELS: Record<number, number> = { 0: 1, 2: 3, 4: 2, 6: 4 };

export function decodePng(buffer: Uint8Array): GrayImage {
  const sig = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];
  if (sig.some((b, i) => buffer[i] !== b)) throw new Error('Not a PNG file');
  const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);

  let width = 0;
  let height = 0;
  let colorType = 0;
  const idat: Uint8Array[] = [];
  for (let pos = 8; pos < buffer.length; ) {
    const length = view.getUint32(pos);
    const type = String.fromCharCode(...buffer.subarray(pos + 4, pos + 8));
    const data = buffer.subarray(pos + 8, pos + 8 + length);
    if (type === 'IHDR') {
      width = view.getUint32(pos + 8);
      height = view.getUint32(pos + 12);
      if (buffer[pos + 16] !== 8) throw new Error('Only 8-bit PNGs are supported');
      colorType = buffer[pos + 17];
      if (buffer[pos + 20] !== 0) throw new Error('Interlaced PNGs are not supported');
    } else if (type === 'IDAT') {
      idat.push(data);
    } else if (type === 'IEND') {
      break;
    }
    pos += 12 + length;
  }

  const channels = CHANNELS[colorType];
  if (!channels) throw new Error(`Unsupported PNG color type ${colorType}`);
  const raw = inflateSync(Buffer.concat(idat));
  const stride = width * channels;
  const pixels = new Uint8Array(height * stride);

  for (let y = 0; y < height; y++) {
    const filter = raw[y * (stride + 1)];
    const line = raw.subarray(y * (stride + 1) + 1, (y + 1) * (stride + 1));
    for (let i = 0; i < stride; i++) {
      const a = i >= channels ? pixels[y * stride + i - channels] : 0;
      const b = y > 0 ? pixels[(y - 1) * stride + i] : 0;
      const c = y > 0 && i >= channels ? pixels[(y - 1) * stride + i - channels] : 0;
      let predictor = 0;
      switch (filter) {
        case 1:
          predictor = a;
          break;
        case 2:
          predictor = b;
          break;
        case 3:
          predictor = (a + b) >> 1;
          break;
        case 4: {
          const p = a + b - c;
          const pa = Math.abs(p - a);
          const pb = Math.abs(p - b);
          const pc = Math.abs(p - c);
          predictor = pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
          break;
        }
      }
      pixels[y * stride + i] = (line[i] + predictor) & 0xff;
    }
  }

  const gray = new Uint8Array(width * height);
  for (let i = 0; i < width * height; i++) {
    const o = i * channels;
    const value = channels >= 3 ? 0.299 * pixels[o] + 0.587 * pixels[o + 1] + 0.114 * pixels[o + 2] : pixels[o];
    // Composite on white when there is an alpha channel
    const alpha = channels === 2 ? pixels[o + 1] : channels === 4 ? pixels[o + 3] : 255;
    gray[i] = Math.round((value * alpha + 255 * (255 - alpha)) / 255);
  }
  return { width, height, gray };
}
