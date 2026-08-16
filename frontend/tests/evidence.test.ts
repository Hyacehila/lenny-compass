import { describe, expect, it } from "vitest";
import { mergeCitationImages } from "../src/App";
import type { Citation } from "../src/types";

function citation(): Citation {
  return {
    evidence_id: "source:one",
    evidence_kind: "source_chunk",
    headings: [],
    text: "Fact",
    metadata: {},
    doc_item_refs: [],
    locator: {
      kind: "document_chunk",
      document_id: "document-one",
      images: [
        { image_id: "one", url: "https://example.com/one.png", caption: "One" },
        { image_id: "two", url: "https://example.com/two.png", caption: "Two" },
      ],
    },
    image_observations: [
      {
        image_id: "one",
        url: "https://example.com/one.png",
        caption: "One",
        observation: "A visible chart",
        inspected: true,
      },
      {
        image_id: "three",
        url: "https://example.com/three.png",
        inspected: false,
        error: "VISION_ERROR",
      },
    ],
  };
}

describe("mergeCitationImages", () => {
  it("keeps every source image and overlays matching vision state", () => {
    const images = mergeCitationImages(citation());

    expect(images.map((image) => image.image_id)).toEqual(["one", "two", "three"]);
    expect(images[0].observation).toBe("A visible chart");
    expect(images[1].url).toBe("https://example.com/two.png");
    expect(images[2].error).toBe("VISION_ERROR");
  });
});
