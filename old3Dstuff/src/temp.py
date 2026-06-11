historical_stopwords = {   
 # Original Base List
    "och", "att", "som", "i", "en", "ett", "den", "det", "de", "dem", "hans", 
    "hon", "han", "hennes", "vi", "ni", "mig", "dig", "sig", "sin", "sitt", 
    "sina", "på", "för", "av", "af", "till", "til", "at", "med", "om", "är", 
    "var", "vara", "varit", "ha", "hade", "har", "här", "hr", "där", "då", 
    "nu", "ut", "in", "upp", "ner", "sedan", "eller", "men", "mot", "mycket", 
    "också", "efter", "uppå", "uti", "honom", "henne", "herr", "kongl", "kong", 
    "kungl", "fr", "ur", "vid", "kan", "skall", "skulle", "vilket", "vilken", 
    "samt", "icke", "ej", "blifwa", "blivit", "bli", "dito", "rummet", "denna",
      "dessa", "så", "något", "några", "första","andra","dennes",
    
    # Archaic Spelling Variations
    "hwilken", "hwilket", "hvilken", "hvilka", "hvilket", "iagh", "iag", "migh", 
    "sigh", "dhe", "dhet", "medh", "effter", "eftter", "uthi", "then", "thet", 
    "thetta", "hwad", "hvad", "ngot", "något", "kunde", "des", "ock", "ifrn", 
    "ifrån", "frn", "från", "hwar", "denne", "fven", "även", "een", "aff", "war",
    "uthan", "angde", "angående", "anno", "dren", "her", "carll", "oluff",
    "jag","jagh", "min", "dhen","dher", "sampt", "the", "dee", "fwer", "nr", "man", 
    "blifvit", "hafwer", "hafva", "emellan", "hos","hoos", "intet", "der", "mera", "mer",
    "äfwen", "äfven", "hwarför", "hvarför", "hvarföre", "hwarföre", "hvarföre", "widare", 
    "widhare", "hwarom", "hvarom", "hwarföre","emedan", "emellertid", "ähr",
    
    # NEW: Added from your 10% frequency check (Procedural & Meta words)
    "samma", "wara", "detta", "såsom", "någon", "genom", "denne", "warit", 
    "måtte", "kunna", "deß", "sådant", "alt", "wäl", "deras", "således", 
    "vice", "dock", "annan", "allenast", "berörde", "wägnar", "alla", 
    "ingen", "kunnat", "wore", "del", "wille", "all", "när", "huru", "annat", "dett","dels",
    "öfver","uppgifvit"
    
    # Common Names (The person-filter)
    "anders", "olof", "lars", "andersson", "carl", "eric", "nils", "erik", 
    "jan", "per", "jansson", "ersson", "erich", "michael", "magnus", "anna", "johan",
    "laron", "anderon", "eron", "larsson","joh","olsson","ang","nilsson","johansson","eriksson",
    "persson", "mårtensson", "gustafsson", "gustafson","and", "jacobsson", "carlsson",
    "ericsson","jacob","daniel","danielsson",
      # OCR butchered "-sson" endings
    "von", # Common in noble names/titles but acts as filler
    
    # Time and Date filler
    "juli", "juni", "mars", "april", "maj", "december", "october", "november",
    "januarii", "februarii", "martii", "dag", "dagh", "åhr", "junii",

        "utslag", "dom", "remiss", "besvär", "upprop", "förbehåll",
    "exception", "exceptio", "duplica", "replica", "libell",
    "insinuerat", "insin", "refererat", "protocoll", "protocoller",
    "dombok", "prot", "rot", "borta", "sessionen", "uprop",
    "arkiv", "liber", "sessionen",

    # Surviving abbreviations
    "ang", "ing", "bil", "bif", "besv", "fullm", "ins",
    "mts", "mfl", "joh", "ans", "rdr", "pgr", "mgr", "uth",
    "stl", "lit", "litt", "feb", "okt", "nov", "dec", "sept",
    "febr", "fol", "cap", "pars", "par", "mtl", "les", "tom",

    # Titles and honorifics
    "wälborne", "högwälborne", "högl", "nådige", "edle", "assessoren", "assessorerne",
    "assessorer", "assessores", "hoffrätten", "hofrätten",
    "hofrätts", "rätts", "notarien", "baron", "rådet", "rådh",
    "herrar", "orden", "excellence", "excells", "eders", "fidem","hof",

    # Month variants
    "octob", "martij", "aprilis",

    # OCR noise
    "muff", "nog", "nogot", "cons","noch",
    "aderton", "adertonde", "adjö", "aldrig", "all", "alla", "allas", "allt", "alltid", "alltså", 
    "andra", "andras", "annan", "annat", "artonde", "att", "av", "bakom", "bara", "behöva", 
    "beslut", "bland", "blev", "bli", "blir", "blivit", "borde", "bort", "borta", "bra", "bäst", 
    "bättre", "båda", "både", "dag", "dagar", "de", "del", "dem", "den", "denna", "deras", 
    "dess", "dessa", "det", "detta", "dig", "din", "dina", "dit", "ditt", "dock", "du", "där", 
    "därför", "då", "efter", "eftersom", "eller", "en", "enligt", "er", "era", "ert", "ett", 
    "fall", "fanns", "fast", "fem", "fick", "fin", "finnas", "finns", "fler", "flera", "flesta", 
    "fram", "från", "fyra", "få", "får", "fått", "följande", "för", "före", "första", "ge", 
    "genom", "ger", "gick", "gjorde", "gjort", "god", "gott", "gälla", "gäller", "gärna", "gå", 
    "gång", "går", "gått", "gör", "göra", "ha", "hade", "haft", "han", "hans", "har", "hela", 
    "heller", "helt", "henne", "hennes", "hit", "hon", "honom", "hur", "här", "hög", "högre", 
    "högst", "i", "ibland", "idag", "igen", "igår", "imorgon", "in", "inför", "inga", "ingen", 
    "inget", "innan", "inne", "inom", "inte", "ja", "jag", "kan", "kanske", "kom", "komma", 
    "kommer", "kommit", "kunde", "kunna", "kvar", "ligga", "ligger", "lika", "lite", "liten", 
    "litet", "lägga", "länge", "man", "med", "mellan", "men", "mer", "mera", "mest", "mig", 
    "min", "mina", "mindre", "minst", "mitt", "mot", "mycket", "många", "måste", "ned", "nej", 
    "ni", "nog", "nu", "nummer", "när", "nästa", "någon", "något", "några", "och", "också", 
    "ofta", "olika", "om", "oss", "på", "redan", "rätt", "sade", "sagt", "samma", "samt", 
    "sedan", "sig", "sin", "sina", "sista", "sitt", "sju", "ska", "skall", "skulle", "som", 
    "stor", "stora", "stort", "står", "större", "störst", "säga", "säger", "sätt", "så", "ta", 
    "tar", "till", "tills", "tre", "två", "under", "upp", "ur", "ut", "utan", "vad", "var", 
    "vara", "varför", "varit", "varje", "vem", "vi", "vid", "vidare", "vilka", "vilken", 
    "vilket", "vill", "visst", "väl", "vår", "våra", "vårt", "än", "ändå", "ännu", "är", 
    "även", "åt", "åtta", "över","saak","dören","egen","egentligen","eck","est","idkas","idka","idkar"

    # HISTORICAL VARIANTS (Pre-1906)
    "af", "utaf", "äfwen", "äfven", "ähr", "hadhe", "hafva", "hafwer", "hafwa", "haffua", 
    "härr", "dhet", "dhen", "dhenne", "dhesse", "dheras", "vthi", "vth", "vthan", "wid", 
    "widh", "war", "wara", "warit", "woro", "hwilken", "hwilka", "hvilken", "hvilket", 
    "ther", "theras", "then", "thet", "thätta", "ehwad", "ehuru", "emedan", "alldenstund",
    "uppå", "opå", "vppå", "uthan", "iämwäl", "jemwäl", "blifwa", "blifwit", "gifwit",
    "finge", "gorde", "giordt", "skola", "skulle", "måst", "måste", "kunde", "kunnat",
    "mott", "emoth", "iemte", "jemte", "tillsammans", "sampt", "eljest", "ellies"

    # COURT/ADMIN NOISE
    "herr", "herrar", "h:r", "hrr", "välborne", "högwälborne", "excellence", "notarien",
    "assessoren", "assessorerne", "protocoll", "acten", "bilaga", "insinuerat", "refererat",
    "ibm", "ibidem", "folio", "fol", "pag", "paginam"
}

# maybe "riddaren", "riddare"

historical_stopwords = {
    # Original Base List
    "och", "att", "som", "i", "en", "ett", "den", "det", "de", "dem", "hans", 
    "hon", "han", "hennes", "vi", "ni", "mig", "dig", "sig", "sin", "sitt", 
    "sina", "på", "för", "av", "af", "till", "til", "at", "med", "om", "är", 
    "var", "vara", "varit", "ha", "hade", "har", "här", "hr", "där", "då", 
    "nu", "ut", "in", "upp", "ner", "sedan", "eller", "men", "mot", "mycket", 
    "också", "efter", "uppå", "uti", "honom", "henne", "herr", "kongl", "kong", 
    "kungl", "fr", "ur", "vid", "kan", "skall", "skulle", "vilket", "vilken", 
    "samt", "icke", "ej", "blifwa", "blivit", "bli", "dito", "rummet",
    
    # Archaic Spelling Variations
    "hwilken", "hwilket", "hvilken", "hvilka", "hvilket", "iagh", "iag", "migh", 
    "sigh", "dhe", "dhet", "medh", "effter", "eftter", "uthi", "then", "thet", 
    "thetta", "hwad", "hvad", "ngot", "något", "kunde", "des", "ock", "ifrn", 
    "ifrån", "frn", "från", "hwar", "denne", "fven", "även", "een", "aff", "war",
    "uthan", "angde", "angående", "anno", "dren", "her", "carll", "oluff",
    "jag", "min", "dhen", "sampt", "the", "dee", "fwer", "nr", "man", 
    "blifvit", "hafwer", "hafva", "emellan", "hos", "intet", "der", "mera", "mer",
    
    # NEW: Added from your 10% frequency check (Procedural & Meta words)
    "samma", "wara", "detta", "såsom", "någon", "genom", "denne", "warit", 
    "måtte", "kunna", "deß", "sådant", "alt", "wäl", "deras", "således", 
    "vice", "dock", "annan", "allenast", "berörde", "wägnar", "alla", 
    "ingen", "kunnat", "wore", "del", "wille", "all", "när", "huru", "annat",
    
    # Common Names (The person-filter)
    "anders", "olof", "lars", "andersson", "carl", "eric", "nils", "erik", 
    "jan", "per", "jansson", "ersson", "erich", "michael", "magnus", "anna", "johan",
    "laron", "anderon", "eron", # OCR butchered "-sson" endings
    "von", # Common in noble names/titles but acts as filler
    
    # Time and Date filler
    "juli", "juni", "mars", "april", "maj", "december", "october", "november",
    "januarii", "februarii", "martii", "dag", "dagh", "åhr", "junii"
}