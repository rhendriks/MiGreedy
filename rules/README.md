# Geohint rules

These files are reference data for extracting locations from CHAOS/NSID records (`--chaos`).

Operators often name a server after where it is — an IATA airport code with a server number
(`lhr18`, `FRA4`), a UN/LOCODE (`nlams`, `uschi`), a city (`amsterdam1`), an internet exchange
(`dns-decix-dns01`), a cloud region (`ap-southeast-1`), or a country / US state (`us39-12`,
`edge-01-ny`). The rules in these files turn such names into places.

Pull requests are welcome.
E.g., if a server name is geolocated wrong, or a place is missed.

The Rust code loads these files.

## File formats

There are two formats. Each file states which one it uses.

### Token list

Whitespace-separated lowercase tokens.

~~~
```text
dns app res srv   # a few examples
web api cdn
```
~~~

### Table

Tabular markdown table.

| code | iata | notes                          |
|------|------|--------------------------------|
| ffm  | fra  | old railway code for Frankfurt |

## The files

| File                                       | Format     | What it holds                                   |
|--------------------------------------------|------------|-------------------------------------------------|
| [`iata-stoplist.md`](iata-stoplist.md)     | token list | three-letter words that are not IATA codes      |
| [`iata-aliases.md`](iata-aliases.md)       | table      | site codes that map to a different IATA code    |
| [`city-stoplist.md`](city-stoplist.md)     | token list | city names that are also everyday server words  |
| [`city-aliases.md`](city-aliases.md)       | table      | local-language city names to the canonical name |
| [`country-aliases.md`](country-aliases.md) | table      | two-letter country tokens that are not ISO 3166 |
| [`exchanges.md`](exchanges.md)             | table      | internet exchange / colo names to their city    |
| [`cloud-regions.md`](cloud-regions.md)     | table      | cloud region codes to their city                |
| [`us-states.md`](us-states.md)             | token list | US state codes                                  |
| [`operators.md`](operators.md)             | sections   | operator-specific site codes, applied by AS     |
| [`metros.md`](metros.md)                   | table      | metro-area codes to their city                  |

These rules provide airports or cities, for which coordinates are in [`../datasets/`](../datasets).
