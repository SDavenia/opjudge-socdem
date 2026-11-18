"""
Map each Habermas Machine situation to its topic and to the higher-level taxonomy of the topics,
and add the topic columns to the data and score CSVs.

The topic of a situation is `question.topic` of the Habermas Machine candidate comparisons
(the same parquet, and the same situation_id = question.id with "S" -> "HB_", as
pluralism/opinion-judge/prepare_data.py): an integer 0-104. TAXONOMY groups those topics into
subdomains and domains.

Writes:
  data/hb_topic_taxonomy.json   the taxonomy: "topics" (id -> label, focus, subdomain, domain)
                                and "domains" (domain -> subdomains -> topic ids)
  data/hb_situation_topics.csv  every Habermas situation with its topic columns
and adds (or refreshes) TOPIC_COLUMNS, right after the situation, in data/hby*.csv and
scores/**/*.csv. It is idempotent: rerun it after adding data or score files.

Usage:
    python topics.py                                         # downloads the parquet (~400 MB)
    python topics.py --parquet hm_all_candidate_comparisons.parquet   # a local copy
"""
import argparse
import json
import logging
import urllib.request
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
SCORES_DIR = ROOT / "scores"
HABERMAS_URL = "https://storage.googleapis.com/habermas_machine/datasets/hm_all_candidate_comparisons.parquet"
TAXONOMY_PATH = DATA_DIR / "hb_topic_taxonomy.json"
SITUATION_TOPICS_PATH = DATA_DIR / "hb_situation_topics.csv"
TOPIC_COLUMNS = ["topic", "topic_label", "topic_subdomain", "topic_domain"]

log = logging.getLogger(__name__)

# (domain, subdomain, [(topic id, label, focus), ...])
TAXONOMY = [
 ("Technology and Digital Society", "AI, Innovation and Science Policy", [
  (0, "Artificial Intelligence & Technology Ethics", "AI risks, ethics, automation, and societal impact."),
  (82, "Science, Research & Innovation Policy", "research funding, space, cloning, scientific priorities"),
  (100, "Global Issues & Technology Impact", "globalisation, internet, society-wide effects")]),
 ("Technology and Digital Society", "Surveillance, Privacy", [
  (24, "National Identity Cards & Surveillance", "ID systems, biometric data, population databases, state monitoring."),
  (56, "Surveillance, Privacy & Security Technology", "facial recognition, CCTV, data tracking, policing technologies.")]),
 ("Technology and Digital Society", "Technology Use", [
  (31, "Technology Use & Public Behaviour (Mobile Phones)", "rules around device usage in public spaces, schools, and transport.")]),
 ("Technology and Digital Society", "Digital Platforms and Media", [
  (17, "BBC & Public Broadcasting Policy", "role, funding, and structure of the BBC."),
  (68, "Media, Social Media & Big Tech Regulation", "data privacy, misinformation, platform control")]),
 ("Environment, Energy and Infrastructure", "Environment & Climate", [
  (23, "Climate Change Policy & Carbon Taxation", "emissions reduction, carbon pricing, environmental targets."),
  (36, "Climate Change Awareness & Urgency", "perception of climate crisis, urgency, and adequacy of current action."),
  (40, "Plastic Use & Environmental Waste Policy", "banning/reducing plastics and environmental impact of waste."),
  (77, "Environment & Government Responsibility", "pollution, conservation, public policy responsibility"),
  (87, "Taxation (Environmental & Consumption Taxes)", "VAT, fuel tax, pollution taxes, targeted taxation")]),
 ("Environment, Energy and Infrastructure", "Energy Policy", [
  (4, "Nuclear Energy & Power Policy", "nuclear power, energy strategy, environmental trade-offs."),
  (39, "Energy Policy (Renewables vs Fossil Fuels)", "energy transition, investment, sustainability strategies."),
  (73, "Energy Policy & Fracking", "fossil fuels, shale gas, environmental trade-offs"),
  (76, "Renewable Energy & Solar Policy", "solar power adoption, subsidies, mandates")]),
 ("Environment, Energy and Infrastructure", "Transportation and Infrastructure", [
  (33, "Transport & Environmental Policy (Fossil Fuel Vehicles)", "banning petrol/diesel cars, transition to electric vehicles."),
  (59, "Public Transport Policy & Accessibility", "cost, funding, and availability of public transportation."),
  (78, "Infrastructure & Transport Development", "railways, roads, megaprojects, national investment"),
  (85, "Transport Infrastructure (Airports & Aviation)", "runways, airports, aviation expansion"),
  (99, "Transport Policy & Urban Mobility", "car bans, traffic laws, autonomous vehicles"),
  (103, "Transport Technology & Environmental Policy", "electric cars, autonomous vehicles, emissions")]),
 ("Health, Healthcare & Bioethics", "Healthcare Systems", [
  (22, "Healthcare System & NHS Policy", "public vs private healthcare, funding, access, and structure of the health system."),
  (48, "Taxation & Public Spending (NHS-focused)", "how to fund public services, especially healthcare."),
  (51, "NHS Performance & Healthcare System Evaluation", "effectiveness, efficiency, and overall quality of the healthcare system.")]),
 ("Health, Healthcare & Bioethics", "Public Health and Lifestyle Policy", [
  (29, "Food Policy, Nutrition & Public Health (Obesity)", "diet regulation, obesity, food industry rules, public health interventions."),
  (88, "Public Health & Food Policy", "sugar tax, obesity, food regulation")]),
 ("Health, Healthcare & Bioethics", "Bioethics", [
  (43, "Genetic Engineering, GMOs & Bioethics", "gene editing, GM foods, pesticides, biotechnology ethics."),
  (61, "Family, Parenting & Reproductive Ethics", "parenting rights, childcare, reproductive technologies, family policy."),
  (72, "Drugs, Medicine & Bioethics", "drug legality, prescriptions, experimental treatments"),
  (83, "Bioethics & Organ Donation", "organ markets, donation systems, medical ethics")]),
 ("Economy, Labour & Welfare", "Taxation & Inequality", [
  (2, "Wealth, Taxation & Inequality", "distribution of wealth, taxation systems, fairness."),
  (71, "Taxation & Economic Policy", "tax rates, inequality, redistribution"),
  (96, "Inequality, Pay & Economic Fairness", "gender pay gap, wealth inequality, wages")]),
 ("Economy, Labour & Welfare", "Labour & Employment", [
  (38, "Minimum Wage & Wage Regulation", "wage floors, living wage, income limits."),
  (58, "Work, Labour Policy & Work-Life Balance", "working hours, flexible work, productivity, labour conditions."),
  (69, "Employment Law & Workers' Rights", "pay, conditions, corporate responsibility, flexibility")]),
 ("Economy, Labour & Welfare", "Welfare & Social Support", [
  (6, "Universal Basic Income (UBI)", "basic income policies and welfare reform."),
  (37, "Housing Policy & Affordability", "housing supply, pricing, rent control, home ownership."),
  (79, "Poverty, Welfare & Social Support", "homelessness, inequality, government assistance")]),
 ("Economy, Labour & Welfare", "Government Spending & Macroeconomics", [
  (11, "Government Intervention & Economic Policy (Subsidies & Industry)", "state support, subsidies, economic intervention."),
  (15, "Public Spending & Budget Priorities", "allocation of government spending (health, defence, education)."),
  (80, "Government Spending & Subsidies", "public funding, arts, regional inequality, subsidies, state support"),
  (91, "Macroeconomics & Austerity", "national debt, austerity, economic policy")]),
 ("Economy, Labour & Welfare", "Economic Systems & Trade", [
  (49, "Political & Economic Systems (Ideologies)", "capitalism vs socialism, democracy, governance models."),
  (81, "Trade, Globalisation & Ethical Consumption", "imports/exports, tariffs, foreign investment, ethical sourcing")]),
 ("Governance, Politics and Public Administration", "General Governance", [
  (1, "UK Domestic Policy & Society (General Public Policy)", "broad UK policy issues—economy, infrastructure, lifestyle, environment."),
  (19, "Public Services & Digital/Physical Infrastructure Access", "access to services (internet, transport, accessibility, public goods)."),
  (34, "Role of Government & Public Services (General Governance)", "state powers, public ownership, welfare, transparency, civil obligations."),
  (50, "Pensions, Ageing & National Service", "retirement age, pensions, elderly welfare, and mandatory service (military/community)."),
  (90, "General Public Policy & Society", "mixed policy issues: pandemics, recycling, divorce, systems"),
  (95, "Social Policy & Cultural Development", "societal values, education, refugees, entrepreneurship"),
  (104, "Government Responsibility & Public Services", "state role in welfare, health, infrastructure, regulation")]),
 ("Governance, Politics and Public Administration", "Elections and Democracy", [
  (8, "Voting Age Reform", "debates specifically about lowering or raising voting age."),
  (14, "Voting Systems & Democratic Participation", "who can vote, compulsory voting, electoral processes."),
  (101, "Electoral Systems & Democracy", "voting systems, democracy reforms")]),
 ("Governance, Politics and Public Administration", "Political Institutions", [
  (30, "Political System Reform & Executive Power", "limits on political leaders, term limits, structure of leadership."),
  (92, "Government Structure & Parliamentary Reform", "House of Lords, MPs, political representation"),
  (74, "Politics, Elections & Political Ethics", "campaign finance, transparency, political conduct")]),
 ("Governance, Politics and Public Administration", "Criminal Justice & Policing", [
  (65, "Criminal Justice System & Punishment", "prisons, sentencing, policing, justice policy."),
  (89, "Policing & Law Enforcement Powers", "police powers, corruption, stop-and-search")]),
 ("Law, Rights and Ethical Frameworks", "Rights & Civil Liberties", [
  (10, "Free Speech, Censorship & Civil Liberties", "speech rights, hate speech, misinformation, censorship."),
  (53, "Individual Rights, Liberties & Social Entitlements", "personal freedoms, legal rights, moral entitlements, and equality.")]),
 ("Law, Rights and Ethical Frameworks", "Legal Restrictions & Responsibilities", [
  (47, "Regulation, Bans & Personal Freedom (General Restrictions)", "broad “should we ban X?” questions across multiple domains."),
  (52, "Limits, Caps & Personal Restrictions (General)", "imposing limits on income, behaviour, consumption, and freedoms."),
  (67, "Age Restrictions & Legal Responsibility", "drinking age, smoking age, adulthood, legal thresholds")]),
 ("Law, Rights and Ethical Frameworks", "Ethics and Philosophy", [
  (18, "Legal & Moral Issues (Law, Rights, Social Norms)", "legality of behaviors, moral/legal boundaries."),
  (55, "General Philosophical & Societal Questions", "broad, mixed, often abstract or speculative societal/technological questions."),
  (57, "Moral Responsibility & Ethics (General)", "moral duties (charity, refugees, global responsibility, religion)."),
  (60, "Everyday Ethics & Social Behaviour", "informal moral dilemmas and day-to-day ethical judgments."),
  (84, "Big Questions (Philosophy, Future & Humanity)", "population, future of humanity, existential questions"),
  (86, "Ethics, Morality & Philosophy", "lying, free will, moral dilemmas, personal ethics")]),
 ("Society, Education & Culture", "Education & Children", [
  (3, "Children, Education & Youth Policy", "upbringing, schooling, child welfare, youth rights."),
  (41, "Child Welfare, Childcare & Family Support", "childcare, school meals, early education, family subsidies."),
  (42, "Education Curriculum & Content", "what is taught in schools (religion, sex education, critical thinking)."),
  (66, "Parenting, Education Rights & Child Welfare", "school choice, homeschooling, parental authority, attendance"),
  (70, "Education System & School Policy", "private vs public schools, uniforms, funding, structure"),
  (93, "Education Policy & Curriculum", "school subjects, inclusion, resources, learning methods")]),
 ("Society, Education & Culture", "Higher Education", [
  (16, "Higher Education Policy & Tuition Fees", "university access, funding, student debt.")]),
 ("Society, Education & Culture", "Culture & Identity", [
  (44, "History, Culture & Collective Memory", "statues, historical figures, national history, cultural identity debates."),
  (64, "General UK Society, Culture & Opinion Questions", "broad evaluative questions about UK society, culture, politics.")]),
 ("Society, Education & Culture", "National Identity & Constitutional Issues", [
  (20, "National Identity, Constitutional Reform & Foreign Policy (General)", "structure of the state, sovereignty, international relations, national identity."),
  (26, "Monarchy vs Republic", "role, relevance, and future of the monarchy."),
  (63, "Devolution, Independence & UK Constitutional Structure", "Scotland, Northern Ireland, regional autonomy, UK unity.")]),
 ("Society, Education & Culture", "Arts & Culture Funding", [
  (45, "Arts Funding & Cultural Policy", "public funding and support for arts and culture.")]),
 ("International Relations & Migration", "Foreign Policy & Aid", [
  (12, "Foreign Aid, International Relations & Humanitarian Policy", "aid, global responsibility, military humanitarian roles")]),
 ("International Relations & Migration", "Immigration & Borders", [
  (13, "Immigration & Asylum Policy", "migration, refugees, border control."),
  (98, "Immigration Policy", "border control, migration levels, citizenship")]),
 ("International Relations & Migration", "Europe & Monetary Policy", [
  (35, "Currency & Monetary Policy (Euro vs Pound)", "national currency decisions and monetary sovereignty."),
  (46, "European Union & Brexit Policy", "EU membership, Brexit consequences, UK–EU relations.")]),
 ("Public Safety, Consumer Protection & Specific Regulations", "Consumer Protection & Advertising", [
  (9, "Regulation of Consumer Goods (Alcohol, Sugar, Tobacco, etc.)", "public health restrictions and consumption laws."),
  (62, "Public Health, Safety & Personal Regulation", "masks, helmets, clothing rules, safety laws"),
  (97, "Advertising, Regulation & Public Health", "junk food ads, political ads, children's advertising")]),
 ("Public Safety, Consumer Protection & Specific Regulations", "Harmful Substance Regulation", [
  (7, "Drug Policy & Substance Regulation", "legality, decriminalisation, and control of drugs."),
  (21, "Sin Taxes & Pricing Policies (Alcohol, Tobacco, Sugar)", "taxation and minimum pricing to influence unhealthy consumption."),
  (25, "Cannabis Policy", "legalisation, decriminalisation, medical vs recreational use."),
  (28, "Tobacco Regulation & Smoking Bans", "banning or restricting cigarettes and tobacco products."),
  (94, "Smoking, Public Health & Personal Freedom", "smoking bans, vaping, public/private rights")]),
 ("Public Safety, Consumer Protection & Specific Regulations", "Weapons & Hazardous Items", [
  (102, "Gun Control & Weapons Policy", "firearms regulation, ownership rights")]),
 ("Public Safety, Consumer Protection & Specific Regulations", "Public Safety", [
  (32, "Fireworks Regulation", "legality and restriction of fireworks.")]),
 ("Animal Welfare", "Animal Welfare", [
  (5, "Animal Ethics & Scientific Testing", "use of animals in research, ethics of experimentation."),
  (27, "Food Ethics, Diet & Animal Consumption", "veganism, meat consumption, ethics of eating animals, sustainability."),
  (54, "Animal Welfare & Protection Policy", "bans on animal products, anti-cruelty laws, conservation."),
  (75, "Animal Rights & Human-Animal Relationships", "pets, zoos, hunting, ethics of animal use")]),
]


def build_taxonomy():
    ids = sorted(t[0] for _, _, topics in TAXONOMY for t in topics)
    assert ids == list(range(105)), f"topic ids are not exactly 0-104: {ids}"
    taxonomy = {"domains": [], "topics": {}}
    for domain, subdomain, topics in TAXONOMY:
        if not taxonomy["domains"] or taxonomy["domains"][-1]["name"] != domain:
            taxonomy["domains"].append({"name": domain, "subdomains": []})
        taxonomy["domains"][-1]["subdomains"].append({"name": subdomain, "topics": [t[0] for t in topics]})
        for topic, label, focus in topics:
            taxonomy["topics"][str(topic)] = {"label": label, "focus": focus.rstrip("."),
                                              "subdomain": subdomain, "domain": domain}
    taxonomy["topics"] = dict(sorted(taxonomy["topics"].items(), key=lambda kv: int(kv[0])))
    return taxonomy


def load_questions(parquet):
    cols = ["question.id", "question.affirming_statement", "question.text", "question.topic"]
    if parquet is None:
        with TemporaryDirectory() as tmpdir:
            parquet = Path(tmpdir) / "candidates.parquet"
            log.info("Downloading %s", HABERMAS_URL)
            urllib.request.urlretrieve(HABERMAS_URL, parquet)
            df = pd.read_parquet(parquet, columns=cols)
    else:
        df = pd.read_parquet(parquet, columns=cols)
    assert (df.groupby("question.id")["question.topic"].nunique() == 1).all(), "a question with several topics"
    return df.drop_duplicates("question.id")


def situation_topics(questions, taxonomy):
    topics = taxonomy["topics"]
    out = pd.DataFrame({
        "situation_id": questions["question.id"].str.replace("S", "HB_", regex=False),
        "situation": questions["question.affirming_statement"],
        "question": questions["question.text"],
        "topic": questions["question.topic"].astype(int),
    })
    for col, key in zip(TOPIC_COLUMNS[1:], ["label", "subdomain", "domain"]):
        out[col] = out["topic"].map(lambda t: topics[str(t)][key])
    return out.sort_values("situation_id").reset_index(drop=True)


def add_topic_columns(path, by_situation):
    """Insert TOPIC_COLUMNS after the situation (or situation_id) column of a CSV."""
    df = pd.read_csv(path, low_memory=False).drop(columns=TOPIC_COLUMNS, errors="ignore")
    if "situation_id" not in df:
        log.info("Skipping %s: no situation_id", path)
        return
    missing = set(df["situation_id"]) - set(by_situation.index)
    if missing:
        raise SystemExit(f"{path}: {len(missing)} situations without a topic, e.g. {sorted(missing)[:3]}")
    anchor = "situation" if "situation" in df else "situation_id"
    at = df.columns.get_loc(anchor) + 1
    topic = by_situation.loc[df["situation_id"], TOPIC_COLUMNS].reset_index(drop=True)
    df = pd.concat([df.iloc[:, :at], topic, df.iloc[:, at:]], axis=1)
    df.to_csv(path, index=False)
    log.info("Added the topic columns to %s (%d rows)", path.relative_to(ROOT), len(df))


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parquet", type=Path, default=None,
                        help="Local copy of hm_all_candidate_comparisons.parquet (default: download it).")
    args = parser.parse_args()

    taxonomy = build_taxonomy()
    TAXONOMY_PATH.write_text(json.dumps(taxonomy, indent=2, ensure_ascii=False) + "\n")
    log.info("Wrote %s (%d topics, %d domains)", TAXONOMY_PATH.relative_to(ROOT), len(taxonomy["topics"]),
             len(taxonomy["domains"]))

    situations = situation_topics(load_questions(args.parquet), taxonomy)
    situations.to_csv(SITUATION_TOPICS_PATH, index=False)
    log.info("Wrote %s (%d situations)", SITUATION_TOPICS_PATH.relative_to(ROOT), len(situations))

    by_situation = situations.set_index("situation_id")
    for path in sorted(DATA_DIR.glob("hby*.csv")) + sorted(SCORES_DIR.glob("**/*.csv")):
        add_topic_columns(path, by_situation)


if __name__ == "__main__":
    main()
