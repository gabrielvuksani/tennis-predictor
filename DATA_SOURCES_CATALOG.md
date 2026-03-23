# Tennis Prediction Data Sources - Comprehensive Catalog

Research date: March 2026

---

## Table of Contents

1. [Free Open Data Repositories](#1-free-open-data-repositories)
2. [Commercial Tennis APIs](#2-commercial-tennis-apis)
3. [Betting Odds APIs & Data](#3-betting-odds-apis--data)
4. [Real-Time / Live Data Feeds](#4-real-time--live-data-feeds)
5. [Scraping Targets](#5-scraping-targets)
6. [Court & Surface Data](#6-court--surface-data)
7. [Weather Data Sources](#7-weather-data-sources)
8. [Social Media & Sentiment Data](#8-social-media--sentiment-data)
9. [Kaggle Datasets](#9-kaggle-datasets)
10. [Open-Source Prediction Tools](#10-open-source-prediction-tools)
11. [News & Injury Data](#11-news--injury-data)
12. [Tournament Structure & Draw Data](#12-tournament-structure--draw-data)
13. [Summary Matrix](#13-summary-matrix)

---

## 1. Free Open Data Repositories

### Jeff Sackmann / Tennis Abstract (BEST FREE SOURCE)

The gold standard for free tennis data. Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license.

| Repository | URL | Data Provided |
|---|---|---|
| **tennis_atp** | https://github.com/JeffSackmann/tennis_atp | ATP rankings, results, match stats (1991-present for tour, 2008+ challengers, 2011+ qualifying) |
| **tennis_wta** | https://github.com/JeffSackmann/tennis_wta | WTA rankings, results, match stats |
| **tennis_MatchChartingProject** | https://github.com/JeffSackmann/tennis_MatchChartingProject | Crowdsourced shot-by-shot data: rally length, shot type, direction, outcome |
| **tennis_slam_pointbypoint** | https://github.com/JeffSackmann/tennis_slam_pointbypoint | Grand Slam point-by-point data (2011-current): serve speed, rally count, serve/return depth, running distance |
| **tennis_pointbypoint** | https://github.com/JeffSackmann/tennis_pointbypoint | Sequential point-by-point data for tens of thousands of pro matches |

**Data fields include:** Match results, player rankings, player ranking points, match statistics (aces, double faults, serve percentages, break points, etc.), tournament metadata (surface, draw size, level).

### tennis-data.co.uk

| Detail | Info |
|---|---|
| **URL** | http://www.tennis-data.co.uk/ |
| **Cost** | Free |
| **Format** | CSV / Excel |
| **Coverage** | Match results (2000+), head-to-head betting odds (2001+) |
| **Data fields** | Court type, surface, tournament round, match results, player rankings/points, set-by-set scores, betting odds from multiple bookmakers |
| **Odds sources** | Bet365, Pinnacle, and others via oddsportal.com |

### TML Database

| Detail | Info |
|---|---|
| **URL** | https://github.com/Tennismylife/TML-Database |
| **Cost** | Free |
| **Description** | Complete and live-updated database with ATP tournament matches |

---

## 2. Commercial Tennis APIs

### Tier 1: Enterprise-Grade Providers

#### Sportradar Tennis API (v3)

| Detail | Info |
|---|---|
| **URL** | https://developer.sportradar.com/tennis/reference/overview |
| **Cost** | Enterprise pricing (contact for quote) |
| **Free trial** | Yes (developer sandbox) |
| **Coverage** | 4,000+ competitions |
| **Feeds** | 37 distinct API feeds |
| **Data** | Real-time scoring, point-by-point, player profiles, rankings, tournament draws, competition schedules, historical results |
| **Format** | JSON |
| **Draw data** | Draws added within 3 hours of official publication for Tier 1 tournaments |

#### SportsDataIO Tennis API

| Detail | Info |
|---|---|
| **URL** | https://sportsdata.io/tennis-api |
| **Cost** | Paid subscription (free trial available) |
| **Data** | Scores, odds, projections, stats, news, images, injuries, standings |
| **Format** | JSON via REST API |
| **Documentation** | https://sportsdata.io/developers/api-documentation/tennis |

#### Enetpulse

| Detail | Info |
|---|---|
| **URL** | https://enetpulse.com/tennis-data/ |
| **Cost** | Paid (contact for pricing) |
| **Data** | Live point-by-point data for all ATP/WTA/Grand Slam matches, player profiles, seasonal stats |
| **Format** | API |

#### Data Sports Group

| Detail | Info |
|---|---|
| **URL** | https://datasportsgroup.com/coverage/tennis/ |
| **Cost** | Paid (free trial available) |
| **Data** | Live point-by-point updates, real-time odds, settlement data |
| **Format** | JSON / XML |

### Tier 2: Mid-Range Providers

#### API-Tennis

| Detail | Info |
|---|---|
| **URL** | https://api-tennis.com/ |
| **Cost** | Paid (14-day free trial) |
| **Data** | Statistics, standings, fixtures, player profiles, odds, WebSocket live events |
| **Coverage** | ATP, WTA, Challenger, ITF |
| **Documentation** | https://api-tennis.com/documentation |

#### Goalserve Tennis API

| Detail | Info |
|---|---|
| **URL** | https://www.goalserve.com/en/sport-data-feeds/tennis-api/prices |
| **Cost** | Paid (30-day free trial) |
| **Data** | Fixtures, live scores, results, in-game player stats, profiles, point-by-point, prematch/inplay odds |
| **Coverage** | Grand Slams, ATP, WTA, ITF |
| **Format** | XML / JSON |

#### SportDevs Tennis API

| Detail | Info |
|---|---|
| **URL** | https://www.sportdevs.com/tennis |
| **Cost** | Free trial (300 requests/day), paid plans after |
| **Data** | Events, livescores, players, teams, predictions, odds, leagues, seasons, statistics, lineups, fixtures, scores, injuries, news, standings |
| **Format** | REST API + WebSockets |

#### Sportbex Tennis API

| Detail | Info |
|---|---|
| **URL** | https://sportbex.com/tennis-api/ |
| **Cost** | Paid |
| **Data** | Live scores, player statistics, betting odds |

### Tier 3: RapidAPI Marketplace Options

| API | URL | Notes |
|---|---|---|
| **Tennis Live Data** | https://rapidapi.com/sportcontentapi/api/tennis-live-data | Free tier available |
| **Tennis API - ATP WTA ITF** | https://rapidapi.com/jjrm365-kIFr3Nx_odV/api/tennis-api-atp-wta-itf | ATP/WTA/ITF coverage |
| **TennisApi** | https://rapidapi.com/fluis.lacasse/api/tennisapi1 | Livescores, rankings, fixtures |
| **Tennis Devs** | https://rapidapi.com/sportdevs-sportdevs-default/api/tennis-devs | Teams, players, standings, odds |
| **Sofascore (unofficial)** | https://rapidapi.com/apidojo/api/sofascore | Unofficial wrapper |

---

## 3. Betting Odds APIs & Data

### The Odds API

| Detail | Info |
|---|---|
| **URL** | https://the-odds-api.com/sports/tennis-odds.html |
| **Cost** | Free tier (500 requests/month), paid plans available |
| **Data** | Match winner odds, game spreads, game totals from multiple bookmakers including Pinnacle |
| **Coverage** | Grand Slams and major tournaments |
| **Format** | JSON REST API |

### Betfair Exchange API & Historical Data

| Detail | Info |
|---|---|
| **URL** | https://developer.betfair.com/ |
| **Historical data** | https://historicdata.betfair.com/ |
| **Cost** | Free API access (Betfair account required) |
| **Historical tiers** | Basic (free, 1-min intervals, no volume) / Advanced (1-sec intervals + volume, paid) / Pro (50ms intervals + volume, paid) |
| **Data since** | 2016 (Stream API era) |
| **Data** | Time-stamped odds and volume, all Exchange markets, filterable by Event ID and market type |
| **Format** | JSON (TAR download files) |
| **Use case** | Backtesting, model calibration, in-play odds analysis |

### Pinnacle Odds Data

| Source | URL | Details |
|---|---|---|
| **Pinnacle Lines API** | https://pinnacleapi.github.io/linesapi | Official API (account required) |
| **Pinnacle Data API (BettingIsCool)** | https://api.bettingiscool.com/ | Years of sharp odds data: opening/closing odds, devigged true probabilities |
| **SportsGameOdds** | https://sportsgameodds.com/pinnacle-odds-api/ | Pre-game, props, live in-play Pinnacle odds |
| **Pinnacle Odds Dropper** | https://www.pinnacleoddsdropper.com/ | Tracks odds movements/drops |

### OddsMatrix

| Detail | Info |
|---|---|
| **URL** | https://oddsmatrix.com/sports/tennis/ |
| **Data** | Pre-match and live odds, push/pull updates, scores, rankings |
| **Format** | XML feeds and data API |

### SportsGameOdds

| Detail | Info |
|---|---|
| **URL** | https://sportsgameodds.com/tennis-betting-odds-api/ |
| **Data** | Money line, over/under, match props, in-game betting |

### BigDataBall

| Detail | Info |
|---|---|
| **URL** | https://www.bigdataball.com/datasets/tennis-data/ |
| **Cost** | Paid |
| **Format** | Excel spreadsheets |
| **Data** | ATP & WTA odds and stats datasets |

---

## 4. Real-Time / Live Data Feeds

| Provider | Protocol | Latency | Coverage | URL |
|---|---|---|---|---|
| **Sportradar** | REST API | Near real-time | 4,000+ competitions | https://developer.sportradar.com/tennis |
| **API-Tennis** | WebSocket | Real-time | ATP/WTA/Challenger/ITF | https://api-tennis.com/ |
| **SportDevs** | REST + WebSocket | Real-time | Multi-tour | https://www.sportdevs.com/tennis |
| **Goalserve** | XML/JSON feed | Real-time | Grand Slams, ATP, WTA, ITF | https://www.goalserve.com/ |
| **Enetpulse** | API | Point-by-point live | ATP/WTA/Grand Slams | https://enetpulse.com/tennis-data/ |
| **Data Sports Group** | JSON/XML | Real-time PBP | ATP/WTA/Grand Slams | https://datasportsgroup.com/ |
| **Betfair Exchange** | Streaming API | ~50ms (Pro tier) | Exchange markets | https://developer.betfair.com/ |

---

## 5. Scraping Targets

### Flashscore

| Detail | Info |
|---|---|
| **URL** | https://www.flashscore.com/ |
| **Data available** | Live scores, match results, point-by-point updates, betting odds, player names/age/DOB/nationality, tournament details, 30+ sports |
| **Scraping tools** | Python + Selenium, BeautifulSoup, Java (Spring Boot + Selenium) |
| **GitHub repos** | `eeghor/flashscore-scraping`, `BrunoMNDantas/FlashscoreAPI`, `panaC/tennis-dataset` |
| **No-code tools** | Apify FlashScore Scraper, ScrapeLead, GetOData |
| **Notes** | JavaScript-heavy site; requires Selenium or similar for full rendering |

### Sofascore

| Detail | Info |
|---|---|
| **URL** | https://www.sofascore.com/tennis |
| **Data available** | Livescores, H2H records, aces, double faults, serve stats, power graphs, ATP/WTA rankings, Grand Slam data |
| **API status** | No public API (restricted by data provider agreements) |
| **Scraping** | Possible but against TOS; unofficial RapidAPI wrapper exists |
| **Widgets** | Available for media partners: https://corporate.sofascore.com/widgets |

### ATP Tour Official Website

| Detail | Info |
|---|---|
| **URL** | https://www.atptour.com/ |
| **Data available** | Player profiles (1000+ players), rankings, tournament results, match statistics, career history |
| **GitHub scrapers** | `glad94/infotennis`, `serve-and-volley/atp-world-tour-tennis-data`, `n63li/Tennis-API`, `beaubellamy/TennisStatistics` |

### WTA Tour Official Website

| Detail | Info |
|---|---|
| **URL** | https://www.wtatennis.com/ |
| **Data available** | Player profiles, rankings, results, stats |

### Tennis Explorer

| Detail | Info |
|---|---|
| **URL** | https://www.tennisexplorer.com/ |
| **Data available** | WTA/ATP stats, rankings, fixtures, results, player H2H, last 10 matches, betting info |

### Ultimate Tennis Statistics

| Detail | Info |
|---|---|
| **URL** | https://www.ultimatetennisstatistics.com/ |
| **Data available** | Elo ratings, surface-specific Elo, match predictions (Crystal Ball), career stats, GOAT rankings |
| **Open source** | Yes (Apache 2.0): https://github.com/mcekovic/tennis-crystal-ball |

### Matchstat.com

| Detail | Info |
|---|---|
| **URL** | https://matchstat.com/ |
| **Data available** | H2H stats, match predictions, live scores, historical data for all Grand Slam/ITF/WTA/ATP matches |
| **API** | Offers a tennis stats API (paid) |

### TennisPrediction.com

| Detail | Info |
|---|---|
| **URL** | https://www.tennisprediction.com/ |
| **Data available** | Match predictions, statistics, detailed match info |

### TennisStats.com

| Detail | Info |
|---|---|
| **URL** | https://tennisstats.com/ |
| **Data available** | Player statistics, performance data |

### Tennis Abstract (Website)

| Detail | Info |
|---|---|
| **URL** | https://www.tennisabstract.com/ |
| **Data available** | Surface speed ratings, player reports, analytical articles |
| **Surface speed** | https://www.tennisabstract.com/reports/atp_surface_speed.html |

---

## 6. Court & Surface Data

### Court Speed Database (courtspeed.com)

| Detail | Info |
|---|---|
| **URL** | https://courtspeed.com/ |
| **Cost** | Free to browse |
| **Data** | Court Pace Index (CPI) measurements 2012-2026 for all major tournaments |
| **Coverage** | Wimbledon, US Open, French Open, Australian Open, Indian Wells, Miami, and all major ATP events |

### Tennis Abstract Surface Speed Ratings

| Detail | Info |
|---|---|
| **URL** | https://www.tennisabstract.com/reports/atp_surface_speed.html |
| **Cost** | Free |
| **Method** | Ace-rate-based rating adjusted for server/returner quality; indirectly accounts for temperature, humidity, wind, balls, surface |
| **Updated** | 2026 ratings available |

### ITF Surface Classification

| Detail | Info |
|---|---|
| **URL** | https://www.itftennis.com/en/about-us/tennis-tech/classified-surfaces/ |
| **Cost** | Free |
| **Data** | Official Court Pace Rating (CPR) for classified surfaces: Category 1 (Slow) through Category 5 (Fast) |
| **Method** | Laboratory-measured coefficient of friction and coefficient of restitution |

### Additional Court Data Sources

| Source | URL | Data |
|---|---|---|
| **Perfect Tennis** | https://www.perfect-tennis.com/court-pace-index-explained/ | CPI explained with Hawk-Eye data |
| **Fiend At Court** | https://fiendatcourt.com/court-pace-index/ | CPI analysis |
| **The Breakpoint (Substack)** | https://thebreakpoint.substack.com/ | Court speed analysis, hard-court homogeneity research |

---

## 7. Weather Data Sources

### Weather APIs for Venue Conditions

| Provider | URL | Cost | Key Features |
|---|---|---|---|
| **OpenWeatherMap** | https://openweathermap.org/api | Free tier (1,000 calls/day) | Current weather, forecasts, historical data; 200,000+ cities; JSON/XML |
| **Open-Meteo** | https://open-meteo.com/ | Free, open-source | Hourly/daily forecasts, historical data, no API key needed |
| **WeatherAPI** | https://www.weatherapi.com/ | Free tier available | Sports venue weather endpoint, 14-day forecasts, historical, air quality |
| **Metcheck Tennis** | https://www.metcheck.com/HOBBIES/tennis.asp | Free | Tennis-specific forecasts by location/club |

### CourtCast AI

| Detail | Info |
|---|---|
| **URL** | https://forecast.tennis/ |
| **Description** | Tennis-specific weather advisor tool |

### Key Weather Variables for Tennis

- Temperature and "feels like" temperature
- Humidity (affects ball bounce and player fatigue)
- Wind speed and direction (affects serve toss, ball trajectory)
- Rain probability (delays, court conditions)
- Altitude (affects ball flight; relevant for venues like Bogota, Denver)
- Wet-bulb globe temperature (heat stress on different surfaces)

---

## 8. Social Media & Sentiment Data

### X (Twitter) Data Sources

| Approach | Details |
|---|---|
| **X API** | Paid tiers; can filter for tennis-related tweets, player mentions, injury reports |
| **Key accounts** | @tennismasterr, tennis tipsters tracked by Matchstat |
| **Sentiment tools** | NLP/AI analysis of fan, bettor, and media posts to gauge public perception |
| **Research basis** | Carnegie Mellon research showed social media sentiment can predict sports outcomes better than traditional methods in some cases |

### Reddit

| Subreddit | Content |
|---|---|
| **/r/tennis** | Match discussion, injury news, player form observations |
| **/r/sportsbook** | Betting lines, value picks, line movement discussion |

### Application to Tennis Prediction

- Social media hype can indicate public money direction (sportsbooks shade lines based on public sentiment)
- Real-time injury/withdrawal news often surfaces on social media before official channels
- Fan sentiment can reveal psychological factors (player confidence, motivation in specific tournaments)
- AI sentiment analysis tools scan thousands of posts to gauge player/match perception

---

## 9. Kaggle Datasets

| Dataset | URL | Description |
|---|---|---|
| **Large ATP and ITF Betting Dataset** | https://www.kaggle.com/datasets/ehallmar/a-large-tennis-dataset-for-atp-and-itf-betting | Large dataset with match results and betting odds |
| **ATP and WTA Tennis Data** | https://www.kaggle.com/datasets/hakeem/atp-and-wta-tennis-data | Results and betting odds from tennis-data.co.uk |
| **ATP Tennis Data with Betting Odds** | https://www.kaggle.com/datasets/edoardoba/atp-tennis-data | ATP data with odds |
| **Tennis Datasets (tag)** | https://www.kaggle.com/datasets?tags=2626-Tennis | All Kaggle datasets tagged "Tennis" |
| **Web Scraping Tennis Statistics** | https://www.kaggle.com/code/dnilim/web-scraping-tennis-statistics | Notebook with scraping code |
| **Beat the Bookmakers with ML** | https://www.kaggle.com/code/edouardthomas/beat-the-bookmakers-with-machine-learning-tennis | ML prediction notebook |

---

## 10. Open-Source Prediction Tools & Models

| Project | URL | Description |
|---|---|---|
| **Tennis Crystal Ball** | https://github.com/mcekovic/tennis-crystal-ball | Full prediction app with Elo, Crystal Ball algorithm (Apache 2.0) |
| **elo-rating-tennis-predictions** | https://github.com/bradklassen/elo-rating-tennis-predictions | Elo-based match prediction in Jupyter |
| **elo_tennis** | https://github.com/hdai/elo_tennis | Elo system applied to men's tennis (Python) |
| **Tennis-Betting-ML** | https://github.com/BrandoPolistirolo/Tennis-Betting-ML | Logistic regression + SGD; 66% accuracy on 125K matches |
| **tennispredictor** | https://github.com/jdlamstein/tennispredictor | ML predictor using Elo + other features |
| **tennis-data (martiningram)** | https://github.com/martiningram/tennis-data | Python tools to scrape tennis data and read datasets |
| **infotennis** | https://github.com/glad94/infotennis | ATP Tour website scraper/processor |
| **atp-world-tour-tennis-data** | https://github.com/serve-and-volley/atp-world-tour-tennis-data | ATP World Tour scraper |
| **R package: deuce** | CRAN / GitHub | Tennis data compilation (used in academic research) |

---

## 11. News & Injury Data

### API Sources with Injury/News Feeds

| Provider | Injury Data | News Feed | URL |
|---|---|---|---|
| **SportsDataIO** | Yes (real-time injury updates) | Yes | https://sportsdata.io/tennis-api |
| **SportDevs** | Yes | Yes | https://www.sportdevs.com/tennis |
| **Sportradar** | Yes (supplementary data) | Yes | https://developer.sportradar.com/tennis |

### Alternative Injury Tracking

| Source | Method | Notes |
|---|---|---|
| **ATP/WTA withdrawal lists** | Scrape official tournament entry/withdrawal pages | Most timely official source |
| **Social media monitoring** | X/Twitter monitoring for player injury mentions | Often faster than official channels |
| **Tennis news sites** | Scrape tennisnerd.net, tennis.com, espn.com/tennis | Contextual injury analysis |
| **Press conference transcripts** | Not available via API; manual monitoring of tennis media | Player quotes about fitness |

---

## 12. Tournament Structure & Draw Data

### Draw/Bracket APIs

| Provider | Draw Data | Seedings | Draw Timing | URL |
|---|---|---|---|---|
| **Sportradar** | Full bracket building API | Yes | Within 3 hours of official publication | https://developer.sportradar.com/tennis/docs/ig-bracket-building |
| **API-Tennis** | Tournament fixtures/draws | Yes | Real-time | https://api-tennis.com/ |
| **SportsDataIO** | Tournament endpoints | Yes | Real-time | https://sportsdata.io/ |
| **Data Sports Group** | Draw data | Yes | Real-time | https://datasportsgroup.com/ |

### Draw Data Fields (Typical)

- Tournament metadata: surface type, draw size, prize money, qualification rounds
- Player seedings and rankings at time of draw
- Bracket positions and match progression
- Byes and qualifiers
- Lucky losers and wildcards

### Free Draw Data

- Jeff Sackmann's repositories include draw round information in match-level data
- ATP/WTA websites publish draws (scrapable)

---

## 13. Summary Matrix

### By Cost Tier

| Tier | Sources | Best For |
|---|---|---|
| **Completely Free** | Jeff Sackmann repos, tennis-data.co.uk, Kaggle, courtspeed.com, ITF surface classification, Tennis Abstract, Open-Meteo, Betfair Basic historical | Historical analysis, model training, backtesting |
| **Freemium (limited free tier)** | The Odds API, SportDevs, RapidAPI tennis APIs, OpenWeatherMap, WeatherAPI | Prototyping, limited real-time data |
| **Free Trial (time-limited)** | API-Tennis (14 days), Goalserve (30 days), Data Sports Group, SportsDataIO | Evaluating data quality before purchase |
| **Paid - Mid Range** | API-Tennis, Goalserve, The Odds API (paid tier), BettingIsCool Pinnacle data | Production models with moderate data needs |
| **Paid - Enterprise** | Sportradar, SportsDataIO, Enetpulse, OddsMatrix, Betfair Advanced/Pro | Full-scale production, real-time feeds, institutional use |

### By Data Type

| Data Type | Best Free Source | Best Paid Source |
|---|---|---|
| **Historical match results** | Jeff Sackmann (tennis_atp / tennis_wta) | Sportradar |
| **Point-by-point** | Jeff Sackmann (slam_pointbypoint) | Enetpulse, Sportradar |
| **Shot-by-shot** | Match Charting Project | Sportradar (limited) |
| **Betting odds (historical)** | tennis-data.co.uk, Betfair Basic | BettingIsCool Pinnacle, Betfair Pro |
| **Live odds** | The Odds API (free tier) | OddsMatrix, Betfair Exchange |
| **Live scores** | Flashscore (scrape) | Sportradar, Goalserve |
| **Player rankings** | Jeff Sackmann | SportsDataIO, Sportradar |
| **Injuries** | Social media / news scraping | SportsDataIO, SportDevs |
| **Court speed** | courtspeed.com, Tennis Abstract | N/A |
| **Weather** | Open-Meteo, OpenWeatherMap | WeatherAPI |
| **Draws/brackets** | Jeff Sackmann (round info) | Sportradar (bracket building API) |
| **Elo ratings** | Ultimate Tennis Statistics | N/A (compute your own) |
| **Surface-specific stats** | Tennis Abstract | Sportradar |
| **Sentiment/social** | Reddit, X (manual) | X API (paid), NLP services |

### Recommended Stack for a Tennis Prediction Model

**Minimum viable (free):**
1. Jeff Sackmann repos - historical results, rankings, match stats
2. tennis-data.co.uk - historical betting odds for calibration
3. courtspeed.com - court speed data
4. Open-Meteo - weather forecasts
5. Flashscore scraping - live scores and near-real-time data

**Production-grade (paid):**
1. Sportradar or Enetpulse - real-time point-by-point
2. Betfair Exchange API - live odds and volume (market-implied probabilities)
3. BettingIsCool Pinnacle data - sharp line calibration
4. The Odds API - multi-bookmaker odds comparison
5. SportsDataIO - injuries and news
6. WeatherAPI - sports venue weather
7. X API - sentiment analysis

---

## Sources

- [API-Tennis](https://api-tennis.com/)
- [SportsDataIO Tennis API](https://sportsdata.io/tennis-api)
- [Sportradar Tennis API](https://developer.sportradar.com/tennis/reference/overview)
- [Goalserve Tennis API](https://www.goalserve.com/en/sport-data-feeds/tennis-api/prices)
- [Data Sports Group Tennis](https://datasportsgroup.com/coverage/tennis/)
- [SportDevs Tennis API](https://www.sportdevs.com/tennis)
- [Enetpulse Tennis Data](https://enetpulse.com/tennis-data/)
- [The Odds API - Tennis](https://the-odds-api.com/sports/tennis-odds.html)
- [Betfair Developer Portal](https://developer.betfair.com/)
- [Betfair Historical Data](https://historicdata.betfair.com/)
- [Pinnacle Lines API](https://pinnacleapi.github.io/linesapi)
- [BettingIsCool Pinnacle Data API](https://api.bettingiscool.com/)
- [OddsMatrix Tennis](https://oddsmatrix.com/sports/tennis/)
- [SportsGameOdds Tennis](https://sportsgameodds.com/tennis-betting-odds-api/)
- [Jeff Sackmann tennis_atp](https://github.com/JeffSackmann/tennis_atp)
- [Jeff Sackmann tennis_wta](https://github.com/JeffSackmann/tennis_wta)
- [Jeff Sackmann Match Charting Project](https://github.com/JeffSackmann/tennis_MatchChartingProject)
- [Jeff Sackmann Slam Point-by-Point](https://github.com/JeffSackmann/tennis_slam_pointbypoint)
- [Jeff Sackmann Point-by-Point](https://github.com/JeffSackmann/tennis_pointbypoint)
- [tennis-data.co.uk](http://www.tennis-data.co.uk/)
- [tennis-data.co.uk data page](http://www.tennis-data.co.uk/data.php)
- [RapidAPI Tennis Collection](https://rapidapi.com/collection/tennis)
- [RapidAPI Tennis Live Data](https://rapidapi.com/sportcontentapi/api/tennis-live-data)
- [Sofascore Tennis](https://www.sofascore.com/tennis)
- [Sofascore API FAQ](https://sofascore.helpscoutdocs.com/article/129-sports-data-api-availability)
- [Flashscore](https://www.flashscore.com/)
- [Flashscore scraping guide (RealDataAPI)](https://www.realdataapi.com/scrape-tennis-match-scores-from-flashscore-website.php)
- [Flashscore scraping (Medium)](https://paulconnollywriter.medium.com/scraping-tennis-player-data-from-flashscore-using-python-and-selenium-ab36c549c762)
- [FlashscoreAPI Java](https://github.com/BrunoMNDantas/FlashscoreAPI)
- [flashscore-scraping Python](https://github.com/eeghor/flashscore-scraping)
- [tennis-dataset (scraper)](https://github.com/panaC/tennis-dataset)
- [Apify Flashscore Scraper](https://apify.com/tomas_jindra/flashscore-scraper)
- [Tennis Explorer](https://www.tennisexplorer.com/)
- [Matchstat.com](https://matchstat.com/)
- [TennisPrediction.com](https://www.tennisprediction.com/)
- [TennisStats.com](https://tennisstats.com/)
- [Tennis Abstract](https://www.tennisabstract.com/)
- [Tennis Abstract Surface Speed Ratings](https://www.tennisabstract.com/reports/atp_surface_speed.html)
- [Court Speed Database (courtspeed.com)](https://courtspeed.com/)
- [ITF Surface Classification](https://www.itftennis.com/en/about-us/tennis-tech/classified-surfaces/)
- [Perfect Tennis CPI](https://www.perfect-tennis.com/court-pace-index-explained/)
- [Fiend At Court CPI](https://fiendatcourt.com/court-pace-index/)
- [The Breakpoint (Substack)](https://thebreakpoint.substack.com/)
- [OpenWeatherMap API](https://openweathermap.org/api)
- [Open-Meteo](https://open-meteo.com/)
- [WeatherAPI](https://www.weatherapi.com/)
- [Metcheck Tennis Weather](https://www.metcheck.com/HOBBIES/tennis.asp)
- [CourtCast AI Weather](https://forecast.tennis/)
- [Ultimate Tennis Statistics](https://www.ultimatetennisstatistics.com/)
- [Tennis Crystal Ball (GitHub)](https://github.com/mcekovic/tennis-crystal-ball)
- [Tennis-API scraper](https://github.com/n63li/Tennis-API)
- [infotennis scraper](https://github.com/glad94/infotennis)
- [atp-world-tour-tennis-data](https://github.com/serve-and-volley/atp-world-tour-tennis-data)
- [TennisStatistics scraper](https://github.com/beaubellamy/TennisStatistics)
- [tennis-data tools (martiningram)](https://github.com/martiningram/tennis-data)
- [Sportbex Tennis API](https://sportbex.com/tennis-api/)
- [BigDataBall Tennis](https://www.bigdataball.com/datasets/tennis-data/)
- [Kaggle Tennis Datasets](https://www.kaggle.com/datasets?tags=2626-Tennis)
- [Kaggle Large Tennis Betting Dataset](https://www.kaggle.com/datasets/ehallmar/a-large-tennis-dataset-for-atp-and-itf-betting)
- [Pinnacle Odds Dropper](https://www.pinnacleoddsdropper.com/)
- [Sportradar Tennis v3 (Postman)](https://www.postman.com/sportradar-media-apis/sportradar-media-apis/documentation/tuyuyee/sportradar-tennis-v3)
- [TML Database](https://github.com/Tennismylife/TML-Database)
- [elo-rating-tennis-predictions](https://github.com/bradklassen/elo-rating-tennis-predictions)
- [elo_tennis](https://github.com/hdai/elo_tennis)
- [Tennis-Betting-ML](https://github.com/BrandoPolistirolo/Tennis-Betting-ML)
- [tennispredictor](https://github.com/jdlamstein/tennispredictor)
- [TheSpread alt-data article](https://www.thespread.com/betting-articles/how-alt-data-like-weather-twitter-sentiment-is-changing-betting-models-the-new-frontier-in-sports-wagering-analytics/)
- [Weather effects on tennis (Nature)](https://www.nature.com/articles/s41598-024-66518-8)
