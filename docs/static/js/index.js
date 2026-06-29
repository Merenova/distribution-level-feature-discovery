window.HELP_IMPROVE_VIDEOJS = false;

const RD_TRACE_IMAGE_VERSION = 'rd-equal-panels-v9';

const GAMMA_TRACE_BY_PORT = {
  49490: {
    gamma: '0.60',
    path: 'static/data/rd/traces/cloze_0097_beta_0_50_gamma_0_60/rd_iteration_trace.json',
  },
  49491: {
    gamma: '0.70',
    path: 'static/data/rd/traces/gamma_0_70/rd_iteration_trace.json',
  },
  49492: {
    gamma: '0.80',
    path: 'static/data/rd/traces/gamma_0_80/rd_iteration_trace.json',
  },
  49493: {
    gamma: '0.90',
    path: 'static/data/rd/traces/gamma_0_90/rd_iteration_trace.json',
  },
  49494: {
    gamma: '0.95',
    path: 'static/data/rd/traces/gamma_0_95/rd_iteration_trace.json',
  },
};

const TRACE_PATH_BY_GAMMA = {
  '0.60': 'static/data/rd/traces/cloze_0097_beta_0_50_gamma_0_60/rd_iteration_trace.json',
};

function togglePanel(group, target) {
  document.querySelectorAll(`[data-panel-group="${group}"]`).forEach((button) => {
    button.classList.toggle('is-active', button.dataset.panelTarget === target);
  });

  document.querySelectorAll(`[data-panel="${group}"]`).forEach((panel) => {
    panel.classList.toggle('is-active', panel.dataset.panelName === target);
  });
}

function markCopied(button, copyText) {
  button.classList.add('copied');
  copyText.textContent = 'Copied';

  window.setTimeout(() => {
    button.classList.remove('copied');
    copyText.textContent = 'Copy';
  }, 2000);
}

function fallbackCopy(text, onDone) {
  const textArea = document.createElement('textarea');
  textArea.value = text;
  textArea.setAttribute('readonly', '');
  textArea.style.position = 'fixed';
  textArea.style.left = '-9999px';
  document.body.appendChild(textArea);
  textArea.select();
  document.execCommand('copy');
  document.body.removeChild(textArea);
  onDone();
}

function copyBibTeX() {
  const bibtexElement = document.getElementById('bibtex-code');
  const button = document.querySelector('.copy-bibtex-btn');
  const copyText = button ? button.querySelector('.copy-text') : null;

  if (!bibtexElement || !button || !copyText) {
    return;
  }

  const text = bibtexElement.textContent.trim();
  const onDone = () => markCopied(button, copyText);

  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(text).then(onDone).catch(() => fallbackCopy(text, onDone));
    return;
  }

  fallbackCopy(text, onDone);
}

function scrollToTop() {
  window.scrollTo({
    top: 0,
    behavior: 'smooth',
  });
}

function formatRdMetric(key, value) {
  if (value === undefined || value === null || Number.isNaN(Number(value))) {
    return '—';
  }

  if (key === 'K') {
    return String(Math.round(Number(value)));
  }

  return Number(value).toFixed(3);
}

function framesForIteration(trace, iteration) {
  return trace.frames.filter((frame) => Number(frame.iteration) === Number(iteration));
}

function versionedTraceImage(imagePath, trace) {
  const version = encodeURIComponent(trace.asset_version || RD_TRACE_IMAGE_VERSION);
  const separator = imagePath.includes('?') ? '&' : '?';
  return `${imagePath}${separator}v=${version}`;
}

function numberSlug(value) {
  if (value === null || value === undefined || String(value).trim() === '') {
    return null;
  }

  const normalizedValue = Number(value);
  if (!Number.isFinite(normalizedValue)) {
    return null;
  }

  return normalizedValue.toFixed(2).replace('.', '_');
}

function prefixSlug(value) {
  if (value === null || value === undefined || String(value).trim() === '') {
    return null;
  }

  const slug = String(value).trim();
  return /^[A-Za-z0-9_-]+$/.test(slug) ? slug : null;
}

function betaGammaToTracePath(prefixValue, betaValue, gammaValue, projectionValue) {
  const betaSlug = numberSlug(betaValue);
  const gammaSlug = numberSlug(gammaValue);
  if (!betaSlug || !gammaSlug) {
    return null;
  }

  const queryPrefix = prefixSlug(prefixValue);
  const projectionSuffixes = {
    tsne: '_tsne',
    umap: '_umap',
  };
  const projectionKey = String(projectionValue || '').toLowerCase();
  const projectionSuffix = projectionSuffixes[projectionKey] || '';
  const traceFolder = queryPrefix
    ? `${queryPrefix}_beta_${betaSlug}_gamma_${gammaSlug}${projectionSuffix}`
    : `beta_${betaSlug}_gamma_${gammaSlug}${projectionSuffix}`;

  return `static/data/rd/traces/${traceFolder}/rd_iteration_trace.json`;
}

function gammaToTracePath(gammaValue) {
  const slug = numberSlug(gammaValue);
  if (!slug) {
    return null;
  }

  const gammaKey = Number(gammaValue).toFixed(2);
  if (TRACE_PATH_BY_GAMMA[gammaKey]) {
    return TRACE_PATH_BY_GAMMA[gammaKey];
  }

  return `static/data/rd/traces/gamma_${slug}/rd_iteration_trace.json`;
}

function resolveRdTracePath(widget) {
  const params = new URLSearchParams(window.location.search);
  const queryPrefix = params.get('prefix');
  const queryBeta = params.get('beta');
  const queryGamma = params.get('gamma');
  const queryProjection = params.get('projection');
  const queryBetaGammaTrace = betaGammaToTracePath(queryPrefix, queryBeta, queryGamma, queryProjection);
  if (queryBetaGammaTrace) {
    return queryBetaGammaTrace;
  }

  const queryTrace = gammaToTracePath(queryGamma);
  if (queryTrace) {
    return queryTrace;
  }

  const portTrace = GAMMA_TRACE_BY_PORT[Number(window.location.port)];
  if (portTrace) {
    return portTrace.path;
  }

  return widget.dataset.rdTrace;
}

function setRdTraceMode(widget, state, trace, mode) {
  state.mode = mode;
  if (mode === 'animation') {
    startRdTraceAnimation(widget, state, trace);
  } else {
    stopRdTraceAnimation(state);
  }
  updateRdTrace(widget, state, trace);
}

function stopRdTraceAnimation(state) {
  if (state.animationTimer) {
    window.clearInterval(state.animationTimer);
    state.animationTimer = null;
  }
}

function startRdTraceAnimation(widget, state, trace) {
  stopRdTraceAnimation(state);
  state.animationTimer = window.setInterval(() => {
    state.animationFrameIndex = (state.animationFrameIndex + 1) % trace.frames.length;
    const frame = trace.frames[state.animationFrameIndex];
    state.iteration = Number(frame.iteration);
    state.stage = frame.stage;
    updateRdTrace(widget, state, trace);
  }, 1150);
}

function activeRdFrame(trace, state) {
  if (state.mode === 'animation') {
    return trace.frames[state.animationFrameIndex] || trace.frames[0];
  }

  const iterationFrames = framesForIteration(trace, state.iteration);
  if (!iterationFrames.length) {
    return trace.frames[0];
  }

  if (!iterationFrames.some((frame) => frame.stage === state.stage)) {
    state.stage = iterationFrames[0].stage;
  }

  return iterationFrames.find((frame) => frame.stage === state.stage) || iterationFrames[0];
}

function updateRdTrace(widget, state, trace) {
  const stillPanel = widget.querySelector('.rd-trace-still-panel');
  const frameImage = widget.querySelector('.rd-trace-frame');
  const slider = widget.querySelector('.rd-trace-slider');
  const iterationLabel = widget.querySelector('[data-rd-iteration-label]');
  const stageTabs = widget.querySelector('.rd-stage-tabs');
  const noteText = widget.querySelector('[data-rd-note]');

  widget.querySelectorAll('.rd-mode-tab').forEach((button) => {
    button.classList.toggle('is-active', button.dataset.rdMode === state.mode);
  });

  if (stillPanel) {
    stillPanel.hidden = state.mode !== 'still';
    stillPanel.classList.toggle('is-active', state.mode === 'still');
  }

  const activeFrame = activeRdFrame(trace, state);
  if (!activeFrame) {
    return;
  }
  state.iteration = Number(activeFrame.iteration);
  state.stage = activeFrame.stage;

  if (frameImage) {
    frameImage.src = versionedTraceImage(activeFrame.image, trace);
    frameImage.alt = `${activeFrame.iteration_label} ${activeFrame.stage_label}: semantic and mechanistic RD cluster views`;
  }

  if (slider) {
    const sliderIndex = Math.max(0, state.iterations.indexOf(Number(activeFrame.iteration)));
    slider.value = String(sliderIndex);
  }

  if (iterationLabel) {
    iterationLabel.textContent = activeFrame.iteration_label;
  }

  if (stageTabs) {
    stageTabs.innerHTML = '';
    const iterationFrames = framesForIteration(trace, activeFrame.iteration);
    iterationFrames.forEach((frame) => {
      const button = document.createElement('button');
      button.className = 'panel-tab rd-stage-tab';
      button.type = 'button';
      button.dataset.rdStage = frame.stage;
      button.textContent = frame.stage_label;
      button.classList.toggle('is-active', frame.stage === state.stage);
      button.addEventListener('click', () => {
        state.stage = frame.stage;
        updateRdTrace(widget, state, trace);
      });
      stageTabs.appendChild(button);
    });
  }

  widget.querySelectorAll('[data-rd-metric]').forEach((metricNode) => {
    const metricKey = metricNode.dataset.rdMetric;
    metricNode.textContent = formatRdMetric(metricKey, activeFrame.metrics[metricKey]);
  });

  if (noteText) {
    noteText.textContent = activeFrame.operation_note;
  }
}

function renderRdTrace(widget, trace) {
  const iterations = [...new Set(trace.frames.map((frame) => Number(frame.iteration)))].sort((a, b) => a - b);
  const firstFrame = trace.frames[0];
  const state = {
    mode: 'animation',
    animationFrameIndex: 0,
    animationTimer: null,
    iteration: firstFrame ? Number(firstFrame.iteration) : iterations[0],
    iterations,
    stage: firstFrame ? firstFrame.stage : 'initial',
  };

  const slider = widget.querySelector('.rd-trace-slider');
  if (slider && iterations.length) {
    slider.min = '0';
    slider.max = String(iterations.length - 1);
    slider.step = '1';
    slider.value = '0';
    slider.addEventListener('input', () => {
      state.iteration = iterations[Number(slider.value)];
      state.animationFrameIndex = trace.frames.findIndex((frame) => Number(frame.iteration) === state.iteration);
      if (state.animationFrameIndex < 0) {
        state.animationFrameIndex = 0;
      }
      updateRdTrace(widget, state, trace);
    });
  }

  widget.querySelectorAll('.rd-mode-tab').forEach((button) => {
    button.addEventListener('click', () => {
      setRdTraceMode(widget, state, trace, button.dataset.rdMode);
    });
  });

  updateRdTrace(widget, state, trace);
  startRdTraceAnimation(widget, state, trace);
}

function initRdTraceWidget() {
  const widget = document.querySelector('[data-rd-trace]');
  if (!widget) {
    return;
  }

  fetch(resolveRdTracePath(widget), { cache: 'no-store' })
    .then((response) => {
      if (!response.ok) {
        throw new Error(`RD trace request failed: ${response.status}`);
      }
      return response.json();
    })
    .then((trace) => renderRdTrace(widget, trace))
    .catch(() => {
      widget.classList.add('rd-trace-load-failed');
    });
}

document.addEventListener('DOMContentLoaded', () => {
  const scrollButton = document.querySelector('.scroll-to-top');

  window.addEventListener('scroll', () => {
    if (!scrollButton) {
      return;
    }

    scrollButton.classList.toggle('visible', window.pageYOffset > 300);
  });

  initRdTraceWidget();
});
