mermaid.initialize({
  startOnLoad: false,
  securityLevel: "strict",
  theme: "default"
});

let mermaidRenderQueue = Promise.resolve();

document$.subscribe(function () {
  mermaidRenderQueue = mermaidRenderQueue.then(async function () {
    const diagrams = document.querySelectorAll(
      ".mermaid:not([data-processed='true']):not([data-mermaid-error='true'])"
    );
    const sources = new Map();
    const targets = [];
    for (const diagram of diagrams) {
      const source = diagram.textContent.trim();
      let target = diagram;
      if (diagram.tagName === "PRE") {
        target = document.createElement("div");
        target.className = diagram.className;
        diagram.replaceWith(target);
      }
      target.replaceChildren(document.createTextNode(source));
      sources.set(target, source);
      targets.push(target);
    }
    if (targets.length === 0) {
      return;
    }
    try {
      await mermaid.run({ nodes: targets });
    } catch (error) {
      for (const target of targets) {
        if (target.dataset.processed === "true") {
          continue;
        }
        target.textContent = sources.get(target);
        target.removeAttribute("data-processed");
        target.dataset.mermaidError = "true";
      }
      console.error("Mermaid rendering failed", error);
    }
  });
});
