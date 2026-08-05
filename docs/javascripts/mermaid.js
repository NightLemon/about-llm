mermaid.initialize({
  startOnLoad: false,
  securityLevel: "strict",
  theme: "default"
});

document$.subscribe(function () {
  mermaid.run({ querySelector: ".mermaid" });
});
