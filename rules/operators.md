# Operator site rules

Specific rules for operators that generic rules misinterpreted.
Ambiguity is unavoidable, this file ensures correct mappings when the rules misclassify locations.

## Format

One `##` section per operator. Under it:

- an `ASNs:` line listing the operator's AS numbers (digits, comma-separated; `AS` prefix optional),
- an optional `Source:` line,
- a table with columns `code`, `city`, `country`.

## CZ.NIC

**ASNs:** 25192

**Source:** NSIDs (`dns-<code>-NN.nic.cz`) checked against stats.nic.cz/dashboard/en/Traffic.html, 2026-10-19

| code      | city      | country |
|-----------|-----------|---------|
| cra       | Prague    | CZ      |
| chile     | Santiago  | CL      |
| cosmonova | Kyiv      | UA      |
| nsw       | Sydney    | AU      |
| sixusa    | Seattle   | US      |

## RcodeZero DNS (r.ns.at)

**ASNs:** 1921

**Source:** the operator, 2026-10-20:
three letter codes based on city names that the rules misinterpret as IATA codes (`tld-all-<code>N`).
Location list at https://www.rcodezero.at/solutions/tld#anycast

| code | city                        | country |
|------|-----------------------------|---------|
| ams  | Amsterdam                   | NL      |
| amx  | Amsterdam                   | NL      |
| atl  | Atlanta                     | US      |
| ber  | Berlin                      | DE      |
| blr  | Bengaluru                   | IN      |
| bru  | Brussels                    | BE      |
| buc  | Bucharest                   | RO      |
| bud  | Budapest                    | HU      |
| chi  | Chicago                     | US      |
| dal  | Dallas                      | US      |
| dbx  | Dubai                       | AE      |
| dfw  | Dallas                      | US      |
| dub  | Dublin                      | IE      |
| fam  | Frankfurt am Main           | DE      |
| ffm  | Frankfurt am Main           | DE      |
| gnf  | Geneve                      | CH      |
| ham  | Hamburg                     | DE      |
| hkg  | Hong Kong                   | HK      |
| hon  | Hong Kong                   | HK      |
| jnb  | Johannesburg                | ZA      |
| joh  | Johannesburg                | ZA      |
| klu  | Klagenfurt am Woerthersee   | AT      |
| lax  | Los Angeles                 | US      |
| lga  | New York City               | US      |
| lhr  | London                      | GB      |
| lis  | Lisbon                      | PT      |
| lju  | Ljubljana                   | SI      |
| lon  | London                      | GB      |
| lux  | Luxembourg                  | LU      |
| mad  | Madrid                      | ES      |
| mel  | Melbourne                   | AU      |
| mia  | Miami                       | US      |
| mil  | Milan                       | IT      |
| mmi  | Miami                       | US      |
| mrs  | Marseille                   | FR      |
| mum  | Mumbai                      | IN      |
| nwr  | Newark                      | US      |
| ord  | Chicago                     | US      |
| par  | Paris                       | FR      |
| prs  | Paris                       | FR      |
| rey  | Reykjavik                   | IS      |
| rig  | Riga                        | LV      |
| rix  | Feldkirch                   | AT      |
| san  | Sandefjord                  | NO      |
| sao  | Sao Paulo                   | BR      |
| sbg  | Salzburg                    | AT      |
| sea  | Seattle                     | US      |
| sgp  | Singapore                   | SG      |
| sin  | Singapore                   | SG      |
| sng  | Singapore                   | SG      |
| sof  | Sofia                       | BG      |
| spl  | Sao Paulo                   | BR      |
| spo  | Sao Paulo                   | BR      |
| stl  | Seattle                     | US      |
| sto  | Stockholm                   | SE      |
| svl  | San Jose                    | US      |
| syy  | Sydney                      | AU      |
| tam  | Tampere                     | FI      |
| tky  | Tokyo                       | JP      |
| tlv  | Tel Aviv                    | IL      |
| tnt  | Toronto                     | CA      |
| tor  | Toronto                     | CA      |
| tyo  | Tokyo                       | JP      |
| vie  | Vienna                      | AT      |
| vnn  | Vienna                      | AT      |
| waw  | Warsaw                      | PL      |
| wie  | Vienna                      | AT      |
| zch  | Zuerich                     | CH      |
