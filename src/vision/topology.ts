// Face-mesh topology: which triangles of the 468-vertex MediaPipe mesh belong to which facial region.
// Regions drive everything that must treat skin, eyes, lips and brows differently.
import { MESH_TRIANGLES } from './meshTriangles';

export const NUM_MESH_VERTS = 468;
export const NUM_LANDMARKS = 478; // 468 mesh vertices + 2 x 5 iris points

export const FACE_OVAL = [
  10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150,
  136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
];
export const LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185];
// The two intermediate vertex rings between the outer and inner lip contours.
export const LIPS_MID = [
  76, 77, 90, 180, 85, 16, 315, 404, 320, 307, 306, 408, 304, 303, 302, 11, 72, 73, 74, 184,
  62, 96, 89, 179, 86, 15, 316, 403, 319, 325, 292, 407, 272, 271, 268, 12, 38, 41, 42, 183,
];
// Inner lip ring as two open chains running from one mouth corner (78) to the other (308).
export const MOUTH_LOWER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308];
export const MOUTH_UPPER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308];
// Eye openings, same convention: lower lid chain and upper lid chain between the two corners.
// "A" is the eye on the image-left of an unmirrored frame (the subject's right eye).
export const EYE_A_LOWER = [33, 7, 163, 144, 145, 153, 154, 155, 133];
export const EYE_A_UPPER = [33, 246, 161, 160, 159, 158, 157, 173, 133];
export const EYE_B_LOWER = [362, 382, 381, 380, 374, 373, 390, 249, 263];
export const EYE_B_UPPER = [362, 398, 384, 385, 386, 387, 388, 466, 263];
export const BROW_A = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46];
export const BROW_B = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285];
export const IRIS_A = { center: 468, ring: [469, 470, 471, 472] };
export const IRIS_B = { center: 473, ring: [474, 475, 476, 477] };

export const NOSE_TIP = 1;
export const CHIN = 152;
export const FOREHEAD_TOP = 10;
export const CHEEK_A = 234; // face-oval extremes at ear height
export const CHEEK_B = 454;

export const Region = { Skin: 0, Eye: 1, Lip: 2, Mouth: 3, Brow: 4 } as const;
export type Region = (typeof Region)[keyof typeof Region];

function strip(lower: number[], upper: number[]): number[] {
  const out: number[] = [];
  for (let i = 0; i + 1 < lower.length; i++) {
    const a = lower[i], b = lower[i + 1], c = upper[i + 1], d = upper[i];
    if (a !== d) out.push(a, b, d);
    if (b !== c) out.push(b, c, d);
  }
  return out;
}

function buildTopology() {
  const lips = new Set([...LIPS_OUTER, ...LIPS_MID, ...MOUTH_LOWER, ...MOUTH_UPPER]);
  const brows = [new Set(BROW_A), new Set(BROW_B)];
  const tris: number[] = [];
  const regions: number[] = [];
  for (let i = 0; i < MESH_TRIANGLES.length; i += 3) {
    const t = [MESH_TRIANGLES[i], MESH_TRIANGLES[i + 1], MESH_TRIANGLES[i + 2]];
    let r: Region = Region.Skin;
    if (t.every((v) => lips.has(v))) r = Region.Lip;
    else if (brows.some((b) => t.every((v) => b.has(v)))) r = Region.Brow;
    tris.push(...t);
    regions.push(r);
  }
  // The canonical mesh leaves the eye openings and the mouth interior as holes; fill them.
  const add = (list: number[], r: Region) => {
    for (let i = 0; i < list.length; i += 3) {
      tris.push(list[i], list[i + 1], list[i + 2]);
      regions.push(r);
    }
  };
  add(strip(EYE_A_LOWER, EYE_A_UPPER), Region.Eye);
  add(strip(EYE_B_LOWER, EYE_B_UPPER), Region.Eye);
  add(strip(MOUTH_LOWER, MOUTH_UPPER), Region.Mouth);
  return { triangles: new Uint16Array(tris), triRegion: new Uint8Array(regions) };
}

const built = buildTopology();
/** All render triangles: the canonical tesselation plus eye and mouth fills. */
export const TRIANGLES = built.triangles;
/** Region of each triangle in TRIANGLES. */
export const TRI_REGION = built.triRegion;
/** Triangles of the canonical tesselation only (used for normals). */
export const SURFACE_TRIANGLES = MESH_TRIANGLES;

/** Vertices that sit on plain facial skin away from features: used to sample skin tone and lighting. */
export const SKIN_SAMPLE_VERTS: Uint16Array = (() => {
  const excluded = new Set<number>([
    ...FACE_OVAL, ...LIPS_OUTER, ...LIPS_MID, ...MOUTH_LOWER, ...MOUTH_UPPER,
    ...EYE_A_LOWER, ...EYE_A_UPPER, ...EYE_B_LOWER, ...EYE_B_UPPER, ...BROW_A, ...BROW_B,
  ]);
  // Also drop the first ring of neighbours around eyes, brows and lips: lashes, lid creases and lip
  // edges would bias a skin-tone sample.
  const featureCore = new Set<number>([
    ...LIPS_OUTER, ...EYE_A_LOWER, ...EYE_A_UPPER, ...EYE_B_LOWER, ...EYE_B_UPPER, ...BROW_A, ...BROW_B,
  ]);
  for (let i = 0; i < MESH_TRIANGLES.length; i += 3) {
    const t = [MESH_TRIANGLES[i], MESH_TRIANGLES[i + 1], MESH_TRIANGLES[i + 2]];
    if (t.some((v) => featureCore.has(v))) t.forEach((v) => excluded.add(v));
  }
  const out: number[] = [];
  for (let v = 0; v < NUM_MESH_VERTS; v++) if (!excluded.has(v)) out.push(v);
  return new Uint16Array(out);
})();
