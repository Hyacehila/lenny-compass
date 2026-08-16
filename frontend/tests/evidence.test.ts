import { describe, expect, it } from "vitest";
import { mergeCitationImages, modeDescription, phaseLabel, readinessLabel } from "../src/App";
import type { Citation, HealthStatus } from "../src/types";

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

describe("Chinese interface labels", () => {
  it("renders readiness states in Chinese", () => {
    expect(readinessLabel(null)).toBe("正在连接");
    expect(readinessLabel({ full_ready: true } as HealthStatus)).toBe("系统就绪");
    expect(readinessLabel({ facts_ready: true, lexical_ready: true } as HealthStatus)).toBe("本地知识库已就绪");
  });

  it("localizes known backend phases even when their message is English", () => {
    expect(phaseLabel("retrieve", "Searching local indexes")).toBe("正在检索本地知识库");
    expect(phaseLabel("agentic", "Selecting evidence")).toBe("智能体正在选择和读取证据");
  });

  it("explains the selected retrieval mode", () => {
    expect(modeDescription("direct")).toContain("响应更快");
    expect(modeDescription("agentic")).toContain("多轮搜索");
    expect(modeDescription("agentic")).toContain("需要更长时间");
  });
});
