import { describe, expect, it } from "vitest";

import { allCamerasChosen, cameraPicksToRequest } from "./BirdsEyeTab";

describe("allCamerasChosen", () => {
  it("false when no cameras are known yet (calibration not picked)", () => {
    expect(allCamerasChosen([])).toBe(false);
  });

  it("false while any slot is still unset", () => {
    expect(allCamerasChosen(["img-1", ""])).toBe(false);
  });

  it("true once every slot has an image", () => {
    expect(allCamerasChosen(["img-1", "img-2"])).toBe(true);
  });
});

describe("cameraPicksToRequest", () => {
  it("pairs each pick with its index", () => {
    expect(cameraPicksToRequest(["img-1", "img-2"])).toEqual([
      { camera_index: 0, image_id: "img-1" },
      { camera_index: 1, image_id: "img-2" },
    ]);
  });

  it("skips unset slots, keeping the index of the ones that are set", () => {
    expect(cameraPicksToRequest(["img-1", "", "img-3"])).toEqual([
      { camera_index: 0, image_id: "img-1" },
      { camera_index: 2, image_id: "img-3" },
    ]);
  });

  it("empty picks yields an empty request", () => {
    expect(cameraPicksToRequest([])).toEqual([]);
  });
});
