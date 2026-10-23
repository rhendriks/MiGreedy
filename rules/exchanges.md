# Internet exchanges & colos

**Format: table** — columns `name`, `city`, `country`, `brand`, `notes` (notes optional).

Internet exchange and colocation names operators put in a server name (`dns-decix-dns01`,
`ams-ix`), mapped to the city they are in.

`brand` identifies IXes with presence in several cities (e.g., amsix, decix).
The home city should be tried as default if nothing else in the record points to a place
(so `decix-mad1` is still Madrid, not Frankfurt).

| name      | city              | country | brand | notes     |
|-----------|-------------------|---------|-------|-----------|
| decix     | Frankfurt am Main | DE      | yes   | DE-CIX    |
| linx      | London            | GB      | yes   |           |
| linxnova  | Ashburn           | US      | no    | LINX NoVA |
| lonap     | London            | GB      | no    |           |
| amsix     | Amsterdam         | NL      | yes   | AMS-IX    |
| nlix      | Amsterdam         | NL      | yes   | NL-ix     |
| franceix  | Paris             | FR      | yes   |           |
| netnod    | Stockholm         | SE      | yes   |           |
| ixbr      | Sao Paulo         | BR      | yes   | IX.br     |
| jpnap     | Tokyo             | JP      | yes   |           |
| jpix      | Tokyo             | JP      | yes   |           |
| bbix      | Tokyo             | JP      | yes   |           |
| vix       | Vienna            | AT      | no    |           |
| nixcz     | Prague            | CZ      | no    | NIX.CZ    |
| nixsk     | Bratislava        | SK      | no    |           |
| cecolo    | Prague            | CZ      | no    |           |
| cesnet    | Prague            | CZ      | no    |           |
| sgix      | Singapore         | SG      | no    |           |
| hkix      | Hong Kong         | HK      | no    |           |
| jinx      | Johannesburg      | ZA      | no    |           |
| napafrica | Johannesburg      | ZA      | yes   |           |
| cinx      | Cape Town         | ZA      | no    |           |
| mix       | Milan             | IT      | no    | MIX-IT    |
| namex     | Rome              | IT      | no    |           |
| topix     | Turin             | IT      | no    |           |
| espanix   | Madrid            | ES      | no    |           |
| bix       | Budapest          | HU      | no    |           |
| inex      | Dublin            | IE      | no    |           |
| mskix     | Moscow            | RU      | yes   | MSK-IX    |
| plix      | Warsaw            | PL      | no    |           |
| thinx     | Warsaw            | PL      | no    |           |
| ficix     | Helsinki          | FI      | no    |           |
| bnix      | Brussels          | BE      | no    |           |
| lucix     | Luxembourg        | LU      | no    |           |
| torix     | Toronto           | CA      | no    |           |
| kinx      | Seoul             | KR      | no    |           |
| mykix     | Kuala Lumpur      | MY      | no    |           |
| cabase    | Buenos Aires      | AR      | yes   |           |
| any2      | Los Angeles       | US      | yes   |           |
| nyiix     | New York City     | US      | no    |           |
| sfmix     | San Francisco     | US      | no    |           |
