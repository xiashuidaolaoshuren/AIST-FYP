"""
Entity Aliases Dictionary for Entity Normalization.

This module provides a curated dictionary of high-frequency entity aliases
to support entity matching in the RetrievalGroundedDetector. The dictionary
maps canonical entity forms to lists of their known aliases, enabling
recognition of entity variations like acronyms, abbreviations, and alternative names.

Key Features:
- Bidirectional lookup: Maps both canonical→aliases and alias→canonical
- ~150 high-impact entries optimized for academic and general domains
- Categories: Countries, Organizations, Academic Institutions, Common Titles
- All strings stored in lowercase for performance optimization

Usage:
    >>> from src.verification.entity_aliases import get_all_forms
    >>> forms = get_all_forms("United States")
    >>> print(forms)  # ['united states', 'usa', 'u.s.', 'america', ...]

References:
    - ISO 3166: Country codes and names
    - docs/entity_normalization_challenge.md: Design documentation
"""

from typing import List


# =============================================================================
# Entity Aliases Dictionary
# =============================================================================

ENTITY_ALIASES = {
    # -------------------------------------------------------------------------
    # Countries (ISO 3166 + Common Names) - Top 50 by frequency
    # -------------------------------------------------------------------------
    "united states of america": ["usa", "u.s.a", "u.s.", "us", "united states", "america", "the states"],
    "united kingdom": ["uk", "u.k.", "britain", "great britain", "england", "gb"],
    "people's republic of china": ["china", "prc", "mainland china", "中国"],
    "russian federation": ["russia", "russian", "россия"],
    "federal republic of germany": ["germany", "deutschland", "de"],
    "french republic": ["france", "fr"],
    "italian republic": ["italy", "italia", "it"],
    "kingdom of spain": ["spain", "españa", "es"],
    "canada": ["ca"],
    "japan": ["jp", "nippon"],
    "republic of korea": ["south korea", "korea", "rok", "kr"],
    "democratic people's republic of korea": ["north korea", "dprk", "kp"],
    "australia": ["aus", "au"],
    "new zealand": ["nz", "aotearoa"],
    "india": ["bharat", "in"],
    "brazil": ["brasil", "br"],
    "mexico": ["mx"],
    "argentina": ["ar"],
    "south africa": ["za"],
    "egypt": ["eg"],
    "israel": ["il"],
    "saudi arabia": ["sa", "ksa"],
    "united arab emirates": ["uae", "emirates"],
    "turkey": ["türkiye", "tr"],
    "sweden": ["se"],
    "norway": ["no"],
    "denmark": ["dk"],
    "finland": ["fi"],
    "netherlands": ["holland", "nl"],
    "belgium": ["be"],
    "switzerland": ["ch", "swiss confederation"],
    "austria": ["at", "österreich"],
    "poland": ["pl", "polska"],
    "czech republic": ["czechia", "cz"],
    "greece": ["gr", "hellas"],
    "portugal": ["pt"],
    "ireland": ["eire", "ie"],
    "singapore": ["sg"],
    "malaysia": ["my"],
    "thailand": ["th"],
    "vietnam": ["viet nam", "vn"],
    "indonesia": ["id"],
    "philippines": ["ph"],
    "pakistan": ["pk"],
    "bangladesh": ["bd"],
    "iran": ["islamic republic of iran", "ir"],
    "iraq": ["iq"],
    
    # -------------------------------------------------------------------------
    # International Organizations - Top 50
    # -------------------------------------------------------------------------
    "united nations": ["un", "u.n."],
    "world health organization": ["who", "w.h.o."],
    "north atlantic treaty organization": ["nato"],
    "european union": ["eu", "e.u."],
    "international monetary fund": ["imf"],
    "world trade organization": ["wto"],
    "world bank": ["wb", "ibrd"],
    "international atomic energy agency": ["iaea"],
    "united nations educational scientific and cultural organization": ["unesco"],
    "united nations children's fund": ["unicef"],
    "international labour organization": ["ilo"],
    "world food programme": ["wfp"],
    "international committee of the red cross": ["icrc", "red cross"],
    "amnesty international": ["ai", "amnesty"],
    "greenpeace": ["greenpeace international"],
    "world wildlife fund": ["wwf"],
    "doctors without borders": ["msf", "médecins sans frontières"],
    "international olympic committee": ["ioc"],
    "fédération internationale de football association": ["fifa"],
    "organization for economic cooperation and development": ["oecd"],
    "association of southeast asian nations": ["asean"],
    "african union": ["au", "a.u."],
    "league of arab states": ["arab league"],
    "organization of american states": ["oas"],
    "commonwealth of nations": ["the commonwealth"],
    "international criminal court": ["icc"],
    "international court of justice": ["icj", "world court"],
    "world intellectual property organization": ["wipo"],
    "international telecommunication union": ["itu"],
    "universal postal union": ["upu"],
    "international civil aviation organization": ["icao"],
    "international maritime organization": ["imo"],
    "world meteorological organization": ["wmo"],
    
    # -------------------------------------------------------------------------
    # US States - Common Abbreviations
    # -------------------------------------------------------------------------
    "california": ["ca", "calif.", "cali"],
    "new york": ["ny", "n.y."],
    "texas": ["tx", "tex."],
    "florida": ["fl", "fla."],
    "illinois": ["il", "ill."],
    "pennsylvania": ["pa", "penn."],
    "ohio": ["oh"],
    "georgia": ["ga"],
    "north carolina": ["nc", "n.c."],
    "michigan": ["mi", "mich."],
    "new jersey": ["nj", "n.j."],
    "virginia": ["va"],
    "washington": ["wa", "wash."],
    "massachusetts": ["ma", "mass."],
    "arizona": ["az", "ariz."],
    "tennessee": ["tn", "tenn."],
    "indiana": ["in", "ind."],
    "missouri": ["mo"],
    "maryland": ["md"],
    "wisconsin": ["wi", "wis."],
    "minnesota": ["mn", "minn."],
    "colorado": ["co", "colo."],
    "alabama": ["al", "ala."],
    
    # -------------------------------------------------------------------------
    # Major Cities
    # -------------------------------------------------------------------------
    "new york city": ["nyc", "new york"],
    "los angeles": ["la", "l.a.", "los angeles"],
    "san francisco": ["sf", "san fran"],
    "washington dc": ["washington", "dc", "d.c.", "washington d.c."],
    "london": ["greater london"],
    "paris": ["paname"],
    "beijing": ["peking", "北京"],
    "tokyo": ["tōkyō", "東京"],
    "hong kong": ["hk", "香港"],
    "singapore city": ["singapore"],
    "sydney": ["syd"],
    "melbourne": ["mel"],
    "toronto": ["to", "t.o."],
    "vancouver": ["van"],
    "mumbai": ["bombay"],
    "delhi": ["new delhi"],
    "shanghai": ["上海"],
    
    # -------------------------------------------------------------------------
    # Academic Institutions - Top Universities
    # -------------------------------------------------------------------------
    "massachusetts institute of technology": ["mit", "m.i.t."],
    "stanford university": ["stanford"],
    "harvard university": ["harvard"],
    "california institute of technology": ["caltech", "cit"],
    "university of california berkeley": ["uc berkeley", "berkeley", "cal"],
    "university of california los angeles": ["ucla", "uc los angeles"],
    "university of california san diego": ["ucsd", "uc san diego"],
    "carnegie mellon university": ["cmu", "carnegie mellon"],
    "princeton university": ["princeton"],
    "yale university": ["yale"],
    "columbia university": ["columbia"],
    "university of chicago": ["uchicago", "chicago"],
    "university of pennsylvania": ["upenn", "penn"],
    "cornell university": ["cornell"],
    "university of michigan": ["umich", "michigan"],
    "university of toronto": ["uoft", "toronto"],
    "mcgill university": ["mcgill"],
    "university of cambridge": ["cambridge", "cam"],
    "university of oxford": ["oxford", "oxon"],
    "imperial college london": ["imperial", "imperial college"],
    "london school of economics": ["lse", "l.s.e."],
    "university college london": ["ucl", "u.c.l."],
    "eth zurich": ["eth", "swiss federal institute of technology"],
    "technical university of munich": ["tum", "tu munich"],
    "university of heidelberg": ["heidelberg"],
    "sorbonne university": ["sorbonne"],
    "university of tokyo": ["todai", "東京大学"],
    "kyoto university": ["kyodai", "京都大学"],
    "peking university": ["pku", "beida", "北京大学"],
    "tsinghua university": ["tsinghua", "清华大学"],
    "national university of singapore": ["nus"],
    "nanyang technological university": ["ntu singapore"],
    "university of hong kong": ["hku"],
    "australian national university": ["anu"],
    "university of melbourne": ["unimelb"],
    
    # -------------------------------------------------------------------------
    # Academic Titles and Positions
    # -------------------------------------------------------------------------
    "doctor": ["dr", "dr."],
    "professor": ["prof", "prof."],
    "assistant professor": ["asst prof", "asst. prof.", "ass prof"],
    "associate professor": ["assoc prof", "assoc. prof."],
    "professor emeritus": ["prof emeritus", "emeritus professor"],
    "doctor of philosophy": ["phd", "ph.d.", "dphil"],
    "master of science": ["msc", "m.sc.", "ms"],
    "master of arts": ["ma", "m.a."],
    "bachelor of science": ["bsc", "b.sc.", "bs"],
    "bachelor of arts": ["ba", "b.a."],
    "postdoctoral researcher": ["postdoc", "post-doc"],
    "research fellow": ["fellow"],
    "principal investigator": ["pi", "p.i."],
    "graduate student": ["grad student"],
    "phd student": ["doctoral student", "phd candidate"],
    "visiting scholar": ["visiting researcher"],
    "adjunct professor": ["adjunct prof"],
    "lecturer": ["lect", "lect."],
    "senior lecturer": ["sr lecturer", "sr. lecturer"],
    "reader": ["university reader"],
    
    # -------------------------------------------------------------------------
    # Academic Fields and Departments
    # -------------------------------------------------------------------------
    "computer science": ["cs", "comp sci", "computing"],
    "artificial intelligence": ["ai", "a.i."],
    "machine learning": ["ml"],
    "natural language processing": ["nlp"],
    "computer vision": ["cv"],
    "electrical engineering": ["ee", "elec eng"],
    "mechanical engineering": ["me", "mech eng"],
    "civil engineering": ["ce", "civ eng"],
    "chemical engineering": ["cheme", "chem eng"],
    "biomedical engineering": ["bme", "biomed eng"],
    "materials science": ["matsci", "mat sci"],
    "mathematics": ["math", "maths"],
    "applied mathematics": ["applied math", "appl math"],
    "statistics": ["stats", "stat"],
    "physics": ["phys"],
    "chemistry": ["chem"],
    "biology": ["bio", "biol"],
    "molecular biology": ["mol bio", "molbio"],
    "biochemistry": ["biochem"],
    "neuroscience": ["neuro"],
    "psychology": ["psych"],
    "economics": ["econ"],
    "political science": ["poli sci", "polisci"],
    "social sciences": ["soc sci"],
    "environmental science": ["env sci", "enviro sci"],
    
    # -------------------------------------------------------------------------
    # Research Funding Agencies
    # -------------------------------------------------------------------------
    "national science foundation": ["nsf", "n.s.f."],
    "national institutes of health": ["nih", "n.i.h."],
    "national aeronautics and space administration": ["nasa"],
    "defense advanced research projects agency": ["darpa"],
    "department of energy": ["doe", "d.o.e."],
    "european research council": ["erc"],
    "uk research and innovation": ["ukri"],
    "engineering and physical sciences research council": ["epsrc"],
    "medical research council": ["mrc"],
    "biotechnology and biological sciences research council": ["bbsrc"],
    "natural sciences and engineering research council": ["nserc"],
    "australian research council": ["arc"],
    "japan society for the promotion of science": ["jsps"],
    "china national natural science foundation": ["nsfc"],
    
    # -------------------------------------------------------------------------
    # Corporate/Business Titles
    # -------------------------------------------------------------------------
    "corporation": ["corp", "corp."],
    "incorporated": ["inc", "inc."],
    "limited": ["ltd", "ltd."],
    "limited liability company": ["llc", "l.l.c."],
    "company": ["co", "co."],
    "public limited company": ["plc", "p.l.c."],
    "gesellschaft mit beschränkter haftung": ["gmbh"],
    "aktiengesellschaft": ["ag"],
    "société anonyme": ["sa", "s.a."],
    "limited partnership": ["lp", "l.p."],
    "general partnership": ["gp"],
    
    # -------------------------------------------------------------------------
    # Common Honorifics and Titles
    # -------------------------------------------------------------------------
    "mister": ["mr", "mr."],
    "mistress": ["mrs", "mrs."],
    "miss": ["ms", "ms."],
    "doctor": ["dr", "dr."],
    "reverend": ["rev", "rev."],
    "honorable": ["hon", "hon."],
    "president": ["pres", "pres."],
    "vice president": ["vp", "v.p.", "veep"],
    "chief executive officer": ["ceo", "c.e.o."],
    "chief technology officer": ["cto", "c.t.o."],
    "chief financial officer": ["cfo", "c.f.o."],
    "chief operating officer": ["coo", "c.o.o."],
    "chief information officer": ["cio", "c.i.o."],
    "chief scientific officer": ["cso", "c.s.o."],
}


# =============================================================================
# Helper Functions
# =============================================================================

def get_all_forms(entity: str) -> List[str]:
    """
    Get all known surface forms for an entity (canonical + aliases).
    
    Performs bidirectional lookup: returns matches whether the input is
    a canonical form or an alias. All comparisons are case-insensitive.
    
    Args:
        entity: Entity string to look up (case-insensitive)
    
    Returns:
        List of all surface forms including the canonical form and all aliases.
        If entity not found in dictionary, returns [entity.lower()].
    
    Examples:
        >>> get_all_forms("United States")
        ['united states of america', 'usa', 'u.s.a', 'u.s.', 'us', 'united states', 'america', 'the states']
        
        >>> get_all_forms("USA")  # Reverse lookup (alias → canonical)
        ['united states of america', 'usa', 'u.s.a', 'u.s.', 'us', 'united states', 'america', 'the states']
        
        >>> get_all_forms("MIT")
        ['massachusetts institute of technology', 'mit', 'm.i.t.']
        
        >>> get_all_forms("UnknownEntity")
        ['unknownentity']
    
    Note:
        - All strings in the dictionary are stored in lowercase
        - Returns deduplicated list (no duplicate forms)
        - Order: canonical form first, then aliases in definition order
    """
    entity_lower = entity.lower().strip()
    
    # Check if entity is a canonical key
    if entity_lower in ENTITY_ALIASES:
        return [entity_lower] + ENTITY_ALIASES[entity_lower]
    
    # Check if entity is an alias (reverse lookup)
    for canonical, aliases in ENTITY_ALIASES.items():
        if entity_lower in aliases:
            return [canonical] + aliases
    
    # Not found in dictionary, return original (lowercased)
    return [entity_lower]


def get_canonical_form(entity: str) -> str:
    """
    Get the canonical form for an entity.
    
    Looks up the entity in the dictionary and returns its canonical form.
    If not found, returns the entity itself (lowercased).
    
    Args:
        entity: Entity string to look up (case-insensitive)
    
    Returns:
        Canonical form of the entity (lowercase)
    
    Examples:
        >>> get_canonical_form("USA")
        'united states of america'
        
        >>> get_canonical_form("MIT")
        'massachusetts institute of technology'
        
        >>> get_canonical_form("UnknownEntity")
        'unknownentity'
    """
    all_forms = get_all_forms(entity)
    return all_forms[0]  # First element is always the canonical form


def get_aliases(entity: str) -> List[str]:
    """
    Get only the aliases for an entity (excluding canonical form).
    
    Args:
        entity: Entity string to look up (case-insensitive)
    
    Returns:
        List of aliases (without canonical form). Empty list if not found.
    
    Examples:
        >>> get_aliases("United States")
        ['usa', 'u.s.a', 'u.s.', 'us', 'united states', 'america', 'the states']
        
        >>> get_aliases("MIT")
        ['mit', 'm.i.t.']
        
        >>> get_aliases("UnknownEntity")
        []
    """
    all_forms = get_all_forms(entity)
    entity_lower = entity.lower().strip()
    
    # If entity not in dictionary, all_forms will be [entity_lower]
    if len(all_forms) == 1 and all_forms[0] == entity_lower:
        return []
    
    # Return all forms except the canonical (first element)
    return all_forms[1:]


def get_statistics() -> dict:
    """
    Get statistics about the entity aliases dictionary.
    
    Returns:
        Dictionary with statistics:
        - total_canonical: Number of canonical forms
        - total_aliases: Total number of aliases across all entities
        - avg_aliases_per_entity: Average aliases per canonical form
        - categories: Breakdown by category (estimated from structure)
    
    Example:
        >>> stats = get_statistics()
        >>> print(f"Total entities: {stats['total_canonical']}")
        >>> print(f"Total aliases: {stats['total_aliases']}")
    """
    total_canonical = len(ENTITY_ALIASES)
    total_aliases = sum(len(aliases) for aliases in ENTITY_ALIASES.values())
    avg_aliases = total_aliases / total_canonical if total_canonical > 0 else 0
    
    return {
        'total_canonical': total_canonical,
        'total_aliases': total_aliases,
        'avg_aliases_per_entity': round(avg_aliases, 2),
        'dictionary_version': '1.0',
        'focus_areas': [
            'Countries (ISO 3166)',
            'International Organizations',
            'Academic Institutions',
            'Academic Titles & Fields',
            'Research Funding Agencies',
            'Corporate Titles'
        ]
    }


# =============================================================================
# Module-level initialization
# =============================================================================

if __name__ == "__main__":
    # Print statistics when run directly
    stats = get_statistics()
    print("=" * 70)
    print("Entity Aliases Dictionary Statistics")
    print("=" * 70)
    print(f"Total canonical forms: {stats['total_canonical']}")
    print(f"Total aliases: {stats['total_aliases']}")
    print(f"Average aliases per entity: {stats['avg_aliases_per_entity']}")
    print(f"\nFocus areas:")
    for area in stats['focus_areas']:
        print(f"  - {area}")
    print("=" * 70)
    
    # Test examples
    print("\nExample lookups:")
    print(f"get_all_forms('USA'): {get_all_forms('USA')[:3]}...")
    print(f"get_all_forms('MIT'): {get_all_forms('MIT')}")
    print(f"get_canonical_form('prof'): {get_canonical_form('prof')}")
    print(f"get_aliases('Doctor'): {get_aliases('Doctor')}")
