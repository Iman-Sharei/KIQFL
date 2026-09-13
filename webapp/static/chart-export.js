(function () {
  function chartDataUrl(chartOrCanvas, backgroundColor) {
    const canvas = chartOrCanvas.canvas || chartOrCanvas;
    const bg = backgroundColor || "#ffffff";
    const exportCanvas = document.createElement("canvas");
    exportCanvas.width = canvas.width;
    exportCanvas.height = canvas.height;
    const ctx = exportCanvas.getContext("2d");
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, exportCanvas.width, exportCanvas.height);
    ctx.drawImage(canvas, 0, 0);
    return exportCanvas.toDataURL("image/png");
  }

  function downloadChartPng(chartOrCanvas, filename, backgroundColor) {
    const link = document.createElement("a");
    link.href = chartDataUrl(chartOrCanvas, backgroundColor);
    link.download = filename;
    link.click();
  }

  window.KiqflChartExport = { chartDataUrl, downloadChartPng };
})();
