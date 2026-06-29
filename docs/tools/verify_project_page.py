#!/usr/bin/env python3
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "index.html"

REQUIRED_SECTIONS = [
    "abstract",
    "challenge",
    "method",
    "cluster-dynamics",
    "causal-validation",
    "mechanism-examples",
    "caveats",
    "BibTeX",
]

REQUIRED_ASSETS = [
    "static/images/rd/cross_sil_kmeans_only_K2-5.png",
    "static/images/rd/cross_silhouette_kmeans_example_tsne.png",
    "static/images/rd/method_overview.png",
    "static/images/rd/method_comparison_summary.png",
    "static/images/steering_demon_wp.png",
    "static/images/wave-particle-duality.jpeg",
]

REQUIRED_DATA = [
    "static/data/rd/cross_silhouette_kmeans_example.json",
    "static/data/rd/traces/cloze_0097_beta_0_50_gamma_0_60/rd_iteration_trace.json",
]

GAMMA_TRACE_MAP = {
    "0.60": {
        "beta": "0.50",
        "port": "49490",
        "path": "static/data/rd/traces/cloze_0097_beta_0_50_gamma_0_60/rd_iteration_trace.json",
    },
    "0.70": {
        "beta": "1.50",
        "port": "49491",
        "path": "static/data/rd/traces/gamma_0_70/rd_iteration_trace.json",
    },
    "0.80": {
        "beta": "1.50",
        "port": "49492",
        "path": "static/data/rd/traces/gamma_0_80/rd_iteration_trace.json",
    },
    "0.90": {
        "beta": "1.50",
        "port": "49493",
        "path": "static/data/rd/traces/gamma_0_90/rd_iteration_trace.json",
    },
    "0.95": {
        "beta": "1.50",
        "port": "49494",
        "path": "static/data/rd/traces/gamma_0_95/rd_iteration_trace.json",
    },
}

FORBIDDEN_TEXT = [
    "PAPER_TITLE",
    "AUTHOR_NAMES",
    "BRIEF_DESCRIPTION",
    "TODO:",
    "Lorem ipsum",
    "YOUR_DOMAIN",
    "YOUR REPO HERE",
]


class ProjectPageParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()
        self.images = []
        self.buttons = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.add(attrs["id"])
        if tag == "img" and "src" in attrs:
            self.images.append(attrs["src"])
        if tag == "button":
            self.buttons.append(attrs)


def fail(message):
    print(f"FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def validate_rd_trace(data_file, data, require_split=True):
    for key in (
        "prefix_id",
        "prefix",
        "beta",
        "gamma",
        "trace_source",
        "selection_metric",
        "selection_scope",
        "candidate_prefixes",
        "stage_detection",
        "projection_views",
        "animation_type",
        "display_sample_indices",
        "same_examples_all_frames",
        "frames",
    ):
        if key not in data:
            fail(f"RD trace metadata key missing from {data_file}: {key}")
    if data["trace_source"] != "instrumented_rd_rerun":
        fail(f"RD trace must come from an instrumented RD rerun: {data_file}")
    if data["stage_detection"] != "logged_step_snapshots":
        fail(f"RD trace must use logged step snapshots: {data_file}")
    if data["projection_views"] != ["semantic", "mechanistic"]:
        fail(f"RD trace must show semantic and mechanistic spaces: {data_file}")
    if data["animation_type"] != "js_timed_frames":
        fail(f"RD trace animation must be JS-controlled so metrics can update: {data_file}")
    if data["same_examples_all_frames"] is not True:
        fail(f"RD trace must keep the same examples in every frame: {data_file}")
    if data["selection_scope"] != "page_featured_candidates":
        fail(f"RD trace should state its selection pool: {data_file}")
    if not data["frames"]:
        fail(f"RD trace must export at least one still frame: {data_file}")
    stages = {frame.get("stage") for frame in data["frames"]}
    required_stages = ("initial", "em", "split") if require_split else ("initial", "em")
    for stage in required_stages:
        if stage not in stages:
            fail(f"RD trace should include a logged {stage} stage: {data_file}")
    trace_indices = data["display_sample_indices"]
    if len(trace_indices) != len(set(trace_indices)):
        fail(f"RD trace display sample indices should be unique: {data_file}")
    for frame in data["frames"]:
        for key in ("iteration", "stage", "stage_label", "image", "metrics", "assignments", "view_labels"):
            if key not in frame:
                fail(f"RD trace frame missing key {key}: {data_file}")
        if frame["view_labels"] != ["Semantic space", "Mechanistic space"]:
            fail(f"RD trace frame should label both projection views: {data_file}")
        if len(frame["assignments"]) == 0:
            fail(f"RD trace frame should log assignments: {data_file}")
        for metric in ("L_RD", "D_e", "D_a", "weighted_distortion", "rate", "K"):
            if metric not in frame["metrics"]:
                fail(f"RD trace frame missing metric {metric}: {data_file}")
        frame_image = ROOT / frame["image"]
        if not frame_image.exists():
            fail(f"missing RD trace frame image: {frame['image']}")
    em_split_pairs = []
    for iteration in sorted({frame["iteration"] for frame in data["frames"]}):
        by_stage = {
            frame["stage"]: frame
            for frame in data["frames"]
            if frame["iteration"] == iteration
        }
        if "em" in by_stage and "split" in by_stage:
            em_split_pairs.append((by_stage["em"], by_stage["split"]))
    if require_split and not em_split_pairs:
        fail(f"RD trace should include a same-iteration EM/Split pair: {data_file}")
    if em_split_pairs and not any(em["assignments"] != split["assignments"] for em, split in em_split_pairs):
        fail(f"EM and Split assignments should differ in the logged trace: {data_file}")
    if em_split_pairs and not any(em["metrics"]["L_RD"] != split["metrics"]["L_RD"] for em, split in em_split_pairs):
        fail(f"EM and Split metrics should differ in the logged trace: {data_file}")


def main():
    if not INDEX.exists():
        fail("index.html is missing")
    html = INDEX.read_text(encoding="utf-8")
    css_path = ROOT / "static/css/index.css"
    css = css_path.read_text(encoding="utf-8") if css_path.exists() else ""
    parser = ProjectPageParser()
    parser.feed(html)

    for section_id in REQUIRED_SECTIONS:
        if section_id not in parser.ids:
            fail(f"missing section id #{section_id}")

    for forbidden in FORBIDDEN_TEXT:
        if forbidden in html:
            fail(f"template placeholder still present: {forbidden}")

    for asset in REQUIRED_ASSETS:
        path = ROOT / asset
        if not path.exists():
            fail(f"missing asset: {asset}")
        if asset not in parser.images and asset not in html:
            fail(f"asset not referenced by index.html: {asset}")

    for data_file in REQUIRED_DATA:
        path = ROOT / data_file
        if not path.exists():
            fail(f"missing data file: {data_file}")
        data = json.loads(path.read_text(encoding="utf-8"))
        if data_file.endswith("cross_silhouette_kmeans_example.json"):
            for key in (
                "prefix_id",
                "n_samples",
                "contrast_score",
                "selection_min_samples",
                "plot_k",
                "cluster_circles",
                "projection_types",
                "primary_projection",
                "display_sample_indices",
                "display_sample_count",
                "subsample_strategy",
                "same_examples_all_panels",
                "cluster_circle_style",
                "dot_style",
                "text_style",
                "figure_note_layout",
                "question_alignment",
                "question_text_style",
                "cluster_label_namespace",
                "cluster_label_layout",
                "cluster_label_boundary",
                "cluster_label_reserved_region",
                "highlighted_label_layout",
                "visible_projection_label",
                "figure_title",
                "panel_titles",
                "panel_score_label",
                "panel_score_label_style",
                "row_gap_style",
                "column_gap_style",
                "kmeans_semantic",
                "kmeans_mechanistic",
            ):
                if key not in data:
                    fail(f"metadata key missing from {data_file}: {key}")
            if int(data["n_samples"]) < int(data["selection_min_samples"]):
                fail(f"selected example does not meet min-sample threshold: {data_file}")
            if int(data["plot_k"]) != 4:
                fail(f"example plot should use fixed K=4: {data_file}")
            if data["cluster_circles"] is not True:
                fail(f"example plot should include cluster circles: {data_file}")
            if sorted(data["projection_types"]) != ["tsne"]:
                fail(f"example plot should include only the t-SNE projection: {data_file}")
            if data["primary_projection"] != "tsne":
                fail(f"t-SNE should be the primary example projection: {data_file}")
            if data["same_examples_all_panels"] is not True:
                fail(f"example plot should reuse the same examples in every panel: {data_file}")
            if data["subsample_strategy"] != "balanced_semantic_mechanistic_intersections":
                fail(f"example plot should use balanced cross-label subsampling: {data_file}")
            if data["cluster_circle_style"] != "large":
                fail(f"cluster circles should use the large style: {data_file}")
            if data["dot_style"] != "large":
                fail(f"dots should use the large style: {data_file}")
            if data["text_style"] != "large":
                fail(f"text should use the large style: {data_file}")
            if data["figure_note_layout"] != "question_only":
                fail(f"figure note should contain only the question: {data_file}")
            if data["question_alignment"] != "center":
                fail(f"question should be center-aligned in the figure note: {data_file}")
            if data["question_text_style"] != "large":
                fail(f"question should use large text in the figure note: {data_file}")
            expected_namespace = {"semantic": "S", "mechanistic": "M"}
            if data["cluster_label_namespace"] != expected_namespace:
                fail(f"cluster labels should use distinct S/M namespaces: {data_file}")
            if data["cluster_label_layout"] != "offset_nonoverlap":
                fail(f"cluster labels should use non-overlapping offsets: {data_file}")
            if data["cluster_label_boundary"] != "high_contrast_box":
                fail(f"cluster labels should have a high-contrast boundary: {data_file}")
            if data["cluster_label_reserved_region"] != "silhouette_annotation":
                fail(f"cluster labels should reserve space around silhouette text: {data_file}")
            if data["highlighted_label_layout"] != "removed":
                fail(f"A/B/C/D highlighted labels should be removed: {data_file}")
            if data["visible_projection_label"] != "omitted":
                fail(f"projection method name should be omitted from visible plot labels: {data_file}")
            if data["figure_title"] != "Semantic vs. Mechanistic K-means Clustering Agreement":
                fail(f"figure title should name semantic/mechanistic clustering agreement: {data_file}")
            expected_panel_titles = {
                "semantic_in_semantic": "Semantic K-means clustering, viewed semantically",
                "semantic_in_mechanistic": "Semantic K-means clustering, viewed mechanistically",
                "mechanistic_in_semantic": "Mechanistic K-means clustering, viewed semantically",
                "mechanistic_in_mechanistic": "Mechanistic K-means clustering, viewed mechanistically",
            }
            if data["panel_titles"] != expected_panel_titles:
                fail(f"panel titles should state row clustering and column view: {data_file}")
            if data["panel_score_label"] != "Silhouette score":
                fail(f"panel score labels should explicitly name silhouette score: {data_file}")
            if data["panel_score_label_style"] != "silhouette_score":
                fail(f"panel score label style should use silhouette score naming: {data_file}")
            if data["row_gap_style"] != "separated":
                fail(f"row gap should clearly separate semantic and mechanistic clustering rows: {data_file}")
            if data["column_gap_style"] != "paired":
                fail(f"column gap should keep each row's two views visually paired: {data_file}")
            display_indices = data["display_sample_indices"]
            if len(display_indices) != len(set(display_indices)):
                fail(f"display sample indices should be unique: {data_file}")
            if len(display_indices) != int(data["display_sample_count"]):
                fail(f"display_sample_count mismatch: {data_file}")
            if data.get("highlighted_examples"):
                fail(f"A/B/C/D highlighted examples should not be exported: {data_file}")
            if int(data["kmeans_semantic"].get("K", 0)) != 4:
                fail(f"semantic example K should be 4: {data_file}")
            if int(data["kmeans_mechanistic"].get("K", 0)) != 4:
                fail(f"mechanistic example K should be 4: {data_file}")
            continue

        if data_file.endswith("rd_iteration_trace.json"):
            validate_rd_trace(data_file, data)
            continue

        fail(f"unrecognized data file validation target: {data_file}")

    for gamma, meta in GAMMA_TRACE_MAP.items():
        trace_path = ROOT / meta["path"]
        if not trace_path.exists():
            fail(f"missing gamma trace for gamma={gamma}: {meta['path']}")
        trace_data = json.loads(trace_path.read_text(encoding="utf-8"))
        validate_rd_trace(meta["path"], trace_data, require_split=(gamma == "0.60"))
        if f"{float(trace_data['gamma']):.2f}" != gamma:
            fail(f"gamma trace has wrong gamma value for {meta['path']}")
        if f"{float(trace_data['beta']):.2f}" != meta["beta"]:
            fail(f"gamma trace has wrong beta value for {meta['path']}")

    if "Shared Semantics, Divergent Mechanisms" not in html:
        fail("paper title is missing")
    if 'href="https://arxiv.org/abs/2606.08236"' not in html:
        fail("paper link should point to the arXiv abstract page")
    if 'href="https://github.com/Merenova/distribution-level-feature-discovery"' not in html:
        fail("code link should point to the public GitHub repository")
    if '<link rel="icon" type="image/jpeg" href="static/images/wave-particle-duality.jpeg">' not in html:
        fail("favicon should use the wave-particle duality image")
    if 'id="knobs"' in html or "Two Knobs: Granularity and View Balance" in html:
        fail("Two Knobs section should not be present on the webpage")
    if 'href="static/css/index.css?v=steering-wp-v1"' not in html:
        fail("stylesheet should be cache-busted after switching to the steering demonstration")
    if '<section class="section cluster-band" id="cluster-dynamics">' not in html:
        fail("cluster dynamics section should explicitly use the white band")
    if '<section class="section causal-band" id="causal-validation">' not in html:
        fail("causal validation section should use a distinct background band")
    if ".cluster-band" not in css or "background: #ffffff" not in css:
        fail("cluster dynamics white band styling is missing")
    if ".causal-band" not in css or "background: #f2f3f5" not in css:
        fail("causal validation background band styling is missing")
    if "Unsupervised Feature Discovery<br>by Aligning Semantics and Mechanisms" not in html:
        fail("publication subtitle should force a line break before the byline")
    if '<span class="author-block">Yonsei University</span><br>' not in html:
        fail("publication venue should start on a new line after Yonsei University")
    if "ICML 2026 Spotlight" not in html:
        fail("publication venue should include Spotlight")
    if "subtitle-line" in html or "subtitle-line" in css:
        fail("publication subtitle should use a hard line break, not CSS-only helper spans")
    if "data-panel-group=\"challenge\"" not in html:
        fail("challenge figure switcher is missing")
    if "Example Clusters" not in html:
        fail("example clustering tab is missing")
    if 'class="panel-tabs challenge-panel-tabs"' not in html:
        fail("challenge tabs should use compact unframed spacing")
    if html.count('class="figure-block challenge-figure"') != 2:
        fail("challenge plot figures should use unframed styling")
    if ".challenge-figure img" not in css or "box-shadow: none" not in css:
        fail("challenge plot image frame styling should be removed")
    if "PCA view" in html or "UMAP view" in html:
        fail("PCA/UMAP projection buttons should not be shown")
    if "t-SNE view" in html:
        fail("projection switcher should be removed when using only t-SNE")
    if "t-SNE projection" in html:
        fail("challenge image alt text should not mention t-SNE")
    if "Semantic sim=" in html or "Mechanism L1=" in html:
        fail("challenge example should not expose raw semantic/mechanism metric labels")
    if "Here, meaning compares what the answer says" in html:
        fail("challenge metric explanation should live in the main body, not a standalone triplet note")
    if "(attribution L1)" in html:
        fail("challenge copy should use mechanistic distance without the attribution L1 parenthetical")
    if "cross_silhouette_kmeans_example_tsne.png" not in html:
        fail("t-SNE example image is missing from the page")
    if 'class="figure-block method-overview-figure"' not in html:
        fail("method overview figure should use its own unframed figure styling")
    if "A/B/C/D" in html:
        fail("A/B/C/D annotation text should not appear on the page")
    if "static/css/fontawesome.all.min.css" in html:
        fail("Font Awesome webfont CSS should not be loaded without local webfont assets")
    if "View mismatch: one clustering view can look clean while the other collapses." not in html:
        fail("example clustering caption is missing")
    if "static/images/rd/rd_iteration_trace.gif" in html:
        fail("RD trace should not use a raw GIF because metrics must update per frame")
    required_rd_trace_copy = [
        "data-rd-trace=\"static/data/rd/traces/cloze_0097_beta_0_50_gamma_0_60/rd_iteration_trace.json\"",
        "Animation",
        "Still",
        "RD objective",
        "Semantic distortion",
        "Mechanistic distortion",
        "Weighted distortion",
        "Rate",
    ]
    for required in required_rd_trace_copy:
        if required not in html:
            fail(f"RD trace copy or markup missing: {required}")
    for hidden in ("Beta 0.50", "Gamma 0.60", "data-rd-selected"):
        if hidden in html:
            fail(f"RD trace metrics should hide beta/gamma display: {hidden}")
    required_reader_copy = [
        "Lay summary",
        "When an AI model answers a question, there is usually more than one reasonable response it could give.",
        "Explaining only one chosen answer can miss these alternatives.",
        "looks across many possible responses",
        "groups the ones that mean similar things and seem to come from similar internal signals",
        "This approach helps people choose what to inspect, compare, and test before digging into the model's inner workings",
        "Who sings a Khmer version of \"You've Got A Friend\" in 1973?",
        '<span class="metric-name">Semantic similarity</span> <span class="metric-value">0.95</span>',
        '<span class="metric-name">Mechanistic distance</span> <span class="metric-value">132</span>',
        '<span class="metric-name">Semantic similarity</span> <span class="metric-value">0.62</span>',
        '<span class="metric-name">Mechanistic distance</span> <span class="metric-value">87</span>',
        '<span class="metric-name">Semantic similarity</span> compares what two continuations say',
        '<span class="metric-name">Mechanistic distance</span> compares their attribution signatures',
        "A1 and A2 are semantically close but mechanistically far apart",
        "A2 and A3 are less semantically similar but mechanistically closer",
        "Each dot is a sampled continuation.",
        "Colors show K-means clusters found in one view",
        "Silhouette is a cluster-cleanliness score",
        "A high native score but low cross-view score",
        "View mismatch:",
        "View mismatch: one clustering view can look clean while the other collapses.",
        "Score interpretation: semantic and mechanistic spaces often disagree",
        "Pipeline: the method samples possible answers, represents each answer in two views, and clusters them into auditable modes.",
        "The output is a set of auditable continuation modes, not a single target answer.",
        "The method works at the distribution level",
        "instead of explaining one chosen answer",
        "weights them by model probability",
        "Semantic embeddings capture what the continuation says",
        "Attribution signatures capture which prefix features support it",
        "keeps a small set of modes only when they reduce semantic and mechanistic distortion enough to justify the added complexity",
        "Iteration view: as clusters split, the same points can become cleaner in both semantic and mechanistic views.",
        "RD clustering treats each sampled continuation as a <strong>weighted point with two views</strong>",
        "<strong>entropy-regularized semantic and attribution distortion</strong>",
        "<em>assigning continuations</em>",
        "<strong>distortion reduction is worth the added rate</strong>",
        "the number of clusters emerges from the objective rather than being chosen in advance.",
        "Causal Validation via Intervention",
        "Steering setup:",
        "Correlation check:",
        "RD shows the clearest monotonic response",
        "semantic-only clusters are near zero",
        "single-continuation directions are weaker",
        "<h3>Result</h3>",
        "Positive correlation means the intervention behaves directionally",
        "RD medoid directions show the most consistent monotonic response",
        "KM-Sem stays close to zero",
        "Single transfers only partially from one randomly chosen continuation",
        "<strong>Implication:</strong> KM-Sem's failure suggests that semantic grouping alone can put continuations with shared semantics but different mechanisms into one cluster",
        "Single's weaker transfer, compared with RD, supports the motivation that one sampled continuation can miss the cluster-level mechanism",
        "Steering asks a causal question about a model",
        "<strong>what changes if we deliberately push one internal signal up or down during inference?</strong>",
        "In Pearl's language, this is an intervention rather than a correlation check.",
        "activation steering does this by <strong>adding, removing, or scaling an internal direction or feature</strong>",
        "Our cluster intervention applies this idea to answer modes.",
        "Can one cluster-level direction move the model toward a whole answer mode, rather than only one sampled continuation?",
        "select the signed top-B attribution features from that source",
        "scale those features at their original prefix position and layer",
        "RD</strong>, which uses the medoid from the joint semantic-and-mechanistic cluster",
        "KM-Sem</strong>, a semantic-only K-means medoid",
        "Single</strong>, one randomly chosen attribution vector from the same RD cluster",
        "Across steering strengths, a useful cluster direction should show a monotonic target-cluster logit change",
        "amplification raises the cluster's relative preference, and suppression lowers it",
        "In the demonstration below, the star marks the source continuation defining the steering direction.",
        "star marks the source continuation defining the steering direction",
        "steering_demon_wp.png",
        "the selected RD medoid direction provides top-B attribution features for activation steering",
        "target-logit changes are measured for cluster members",
        "Cluster-Level Transfer Case Study",
        "Who pays for the renovations on <em>Holmes Next Generation</em>?",
        "Mode 1: Production Funding",
        "Mode 2: Not Recognized",
        "RD-medoid source",
        "Single random source",
        "...funded through the production budget...",
        "...spinoff of the original Holmes on Homes series...",
        "...not a widely recognized or officially documented show...",
        "...not a well-known or officially recognized show...",
        "<dd>+0.598</dd>",
        "<dd>100%</dd>",
        "<dd>-0.189</dd>",
        "<dd>+0.239</dd>",
        "<dd>-0.228</dd>",
        "<dd>-0.011</dd>",
        "<dd>+0.258</dd>",
        "Case-study implication:",
        "the RD medoid transfers to held-out continuations in the same mode",
        "a random single continuation can be weak or even reverse sign",
        "captures a cluster-level causal factor shared across continuations",
        "Caveats",
        "Our work treats RD clusters",
        "useful mechanistic hypotheses",
        "not complete circuit explanations",
        "the intervention target is the mean demeaned logit over a sampled continuation",
        "differs from conventional next-token circuit analysis",
        "traces a single target-token logit",
        "the method needs semantic embeddings and mechanistic attribution summaries",
        "clusters can mix answer identity, wording style, hedging, or formatting",
        "steering tests give interventional evidence for cluster-level feature directions",
        "do not trace every token-level pathway to a final token prediction",
        "rate-distortion",
        "attribution signature",
        "semantic embedding",
        "RD objective",
    ]
    for required in required_reader_copy:
        if required not in html:
            fail(f"reader-facing copy missing: {required}")
    if "What to notice:" in html or "Takeaway:" in html:
        fail("old repeated caption lead should be replaced with figure-specific labels")
    if "RD is stronger because its medoid is selected from continuations" in html:
        fail("steering implication should not include the removed RD medoid sentence")
    if 'class="results-strip"' in html:
        fail("Causal Validation should use paragraph introduction, not the removed three-card results strip")
    removed_steering_grid_refs = [
        'data-panel-group="steering-source-projection"',
        "steering_source_cluster5_grid.png",
        "steering_source_cluster5_grid_tsne.png",
        "steering_demonstration.png",
    ]
    for removed in removed_steering_grid_refs:
        if removed in html:
            fail(f"removed PCA/t-SNE steering-source figure should not be referenced: {removed}")
    for label in (
        "View mismatch:",
        "Score interpretation:",
        "Pipeline:",
        "Iteration view:",
        "Steering setup:",
        "Correlation check:",
        "Case-study implication:",
    ):
        if label not in html:
            fail(f"figure-specific caption label missing: {label}")
    removed_case_study_refs = [
        "Mechanism and Circuit Examples",
        "Feature Signature Caveat",
        "overall_step_pair_abs_rho_s.png",
        "Reasoning validation:",
    ]
    for removed in removed_case_study_refs:
        if removed in html:
            fail(f"old mechanism examples content should not be present: {removed}")
    required_method_css = [
        ".method-overview-figure",
        ".method-overview-figure img",
        ".caption-lead",
        "box-shadow: none",
        "background: transparent",
    ]
    for required in required_method_css:
        if required not in css:
            fail(f"method overview unframed styling missing: {required}")
    required_steering_source_css = [
        ".steering-source-copy",
        ".steering-demo-figure",
        ".steering-result-layout",
        ".steering-result-figure",
        ".steering-result-summary",
        ".steering-result-copy",
        ".case-study-prompt",
        ".case-study-grid",
        ".case-study-card",
        ".case-study-row",
        ".case-study-implication",
        ".caveat-band",
        ".caveat-panel",
    ]
    for required in required_steering_source_css:
        if required not in css:
            fail(f"steering source plot styling missing: {required}")
    js_path = ROOT / "static/js/index.js"
    js = js_path.read_text(encoding="utf-8") if js_path.exists() else ""
    if "togglePanel" not in html and "togglePanel" not in js:
        fail("togglePanel helper is missing")
    required_rd_trace_js = [
        "initRdTraceWidget",
        "renderRdTrace",
        "setRdTraceMode",
        "startRdTraceAnimation",
        "stopRdTraceAnimation",
        "animationFrameIndex",
        "GAMMA_TRACE_BY_PORT",
        "TRACE_PATH_BY_GAMMA",
        "'0.60': 'static/data/rd/traces/cloze_0097_beta_0_50_gamma_0_60/rd_iteration_trace.json'",
        "resolveRdTracePath",
        "gamma",
        "rd-trace-slider",
        "rd-stage-tab",
    ]
    for required in required_rd_trace_js:
        if required not in js:
            fail(f"RD trace JavaScript missing: {required}")
    required_bibtex = [
        "@inproceedings{\ncho2026shared,",
        "author={Hyunjin Cho and Youngji Roh and Jaehyung Kim}",
        "booktitle={Forty-third International Conference on Machine Learning}",
        "url={https://openreview.net/forum?id=C9AhjL8aUZ}",
    ]
    for required in required_bibtex:
        if required not in html:
            fail(f"BibTeX entry is missing required content: {required}")

    print("OK: project page structure, assets, and citation verified")


if __name__ == "__main__":
    main()
