import pandas as pd
import csv
import os
from datetime import datetime, timedelta
#2459995.723424_Tautenburg___1652-6323kHz___10-20.3s___sr.wav

species_mapping = [
  '00', 'unidentified', 'Unbestimmt',
  '01', 'unidentified', 'Unbestimmt',
  '02', 'unidentified', 'Unbestimmt',
  '03', 'unidentified', 'Unbestimmt',
  '04', 'unidentified', 'Unbestimmt',
  '05', 'unidentified', 'Unbestimmt',
  '0i', 'unidentified Insect', 'Unbestimmtes Insekt',
  '0b', 'unidentified bat', 'Unbestimmte Fledermaus',
  '0s', 'unidentified mammal', 'Unbestimmter Säuger',
  'w0', 'Wingbeats', 'Flügelschläge',
  'g0', 'goose spec.', 'Gans unbestimmt',
  'kr', 'Common Crane', 'Kranich',
  'zd', 'Little Bittern', 'Zwergdommel',
  'na', 'night heron', 'Nachtreiher',
  'py', 'Pygmy Owl', 'Sperlingskauz',
  'eo', 'Eagle Owl', 'Uhu',
  'ob', 'Ortolan Bunting', 'Ortolan',
  'rx', 'Caspian Tern', 'Raubseeschwalbe',
  'ac', 'Arctic Skua', 'Schmarotzerraubmöwe',
  'ae', 'Arctic Tern', 'Küstenseeschwalbe',
  'av', 'Avocet', 'Säbelschnäbler',
  'bo', 'Barn Owl', 'Schleiereule',
  'by', 'Barnacle Goose', 'Weißwangengans',
  'ba', 'Bar-tailed Godwit', 'Pfuhlschnepfe',
  'br', 'Bearded Tit', 'Bartmeise',
  'bs', 'Berwicks Swan', 'Zwergschwan',
  'bi', 'Bittern', 'Rohrdommel',
  'bk', 'Black Grouse', 'Birkhuhn',
  'ty', 'Black Guillemot', 'Gryllteiste',
  'bx', 'Black Redstart', 'Hausrotschwanz',
  'bj', 'Black Tern', 'Trauerseeschwalbe',
  'b.', 'Blackbird', 'Amsel',
  'bc', 'Blackcap', 'Mönchsgrasmücke',
  'bh', 'Black-headed Gull', 'Lachmöwe',
  'bn', 'Black-necked Grebe', 'Schwarzhalstaucher',
  'bw', 'Black-tailed Godwit', 'Uferschnepfe',
  'bv', 'Black-throated Diver', 'Prachttaucher',
  'bt', 'Blue Tit', 'Blaumeise',
  'bu', 'Bluethroat', 'Blaukehlchen',
  'bl', 'Brambling', 'Bergfink',
  'bg', 'Brent Goose', 'Ringelgans',
  'bf', 'Bullfinch', 'Gimpel',
  'bz', 'Buzzard', 'Mäusebussard',
  'cg', 'Canada Goose', 'Kanadagans',
  'cp', 'Capercaillie', 'Auerhuhn',
  'c.', 'Carrion Crow', 'Rabenkrähe',
  'cw', 'Cettis Warbler', 'Seidensänger',
  'ch', 'Chaffinch', 'Buchfink',
  'cc', 'Chiffchaff', 'Zilpzalp',
  'cf', 'Chough', 'Alpendohle',
  'cl', 'Cirl Bunting', 'Zaunammer',
  'ct', 'Coal Tit', 'Tannenmeise',
  'cd', 'Collared Dove', 'Türkentaube',
  'cm', 'Common Gull', 'Sturmmöwe',
  'cs', 'Common Sandpiper', 'Flussuferläufer',
  'cx', 'Common Scoter', 'Trauerente',
  'cn', 'Common Tern', 'Flussseeschwalbe',
  'co', 'Coot', 'Blässhuhn',
  'ca', 'Cormorant', 'Kormoran',
  'cb', 'Corn Bunting', 'Grauammer',
  'ce', 'Corncrake', 'Wachtelkönig',
  'ci', 'Crested Tit', 'Haubenmeise',
  'cr', 'Common Crossbill', 'Fichtenkreuzschnabel',
  'ck', 'Cuckoo', 'Kuckuck',
  'cu', 'Curlew', 'Brachvogel',
  'dw', 'Dartford Warbler', 'Provencegrasmücke',
  'di', 'Dipper', 'Wasseramsel',
  'do', 'Dotterel', 'Mornellregenpfeifer',
  'dn', 'Dunlin', 'Alpenstrandläufer',
  'd.', 'Dunnock', 'Heckenbraunelle',
  'eg', 'Egyptian Goose', 'Nilgans',
  'e.', 'Eider', 'Eiderente',
  'fp', 'Feral Pigeon', 'Straßentaube',
  'zl', 'Feral/hybrid goose', 'Haus-/Hybridgans',
  'zf', 'Feral/hybrid mallard type', 'Haus/hybrider Stockententyp',
  'ff', 'Fieldfare', 'Wacholderdrossel',
  'fc', 'Firecrest', 'Sommergoldhähnchen',
  'f.', 'Fulmar', 'Eissturmvogel',
  'ga', 'Gadwall', 'Schnatterente',
  'gx', 'Gannet', 'Basstölpel',
  'gw', 'Garden Warbler', 'Gartengrasmücke',
  'gy', 'Garganey', 'Knäkente',
  'gc', 'Goldcrest', 'Wintergoldhähnchen',
  'ea', 'Golden Eagle', 'Steinadler',
  'ol', 'Golden Oriole', 'Pirol',
  'gf', 'Golden Pheasant', 'Goldfasan',
  'gp', 'Golden Plover', 'Goldregenpfeifer',
  'gn', 'Goldeneye', 'Schellente',
  'go', 'Goldfinch', 'Stieglitz',
  'gd', 'Goosander', 'Gänsesäger',
  'gi', 'Goshawk', 'Habicht',
  'gh', 'Grasshopper Warbler', 'Feldschwirl',
  'gb', 'Great Black-backed Gull', 'Mantelmöwe',
  'gg', 'Great Crested Grebe', 'Haubentaucher',
  'nd', 'Great Northern Diver', 'Großer Nordtaucher',
  'nx', 'Great Skua', 'Skua',
  'gs', 'Great Spotted Woodpecker', 'Buntspecht',
  'gt', 'Great Tit', 'Kohlmeise',
  'ge', 'Green Sandpiper', 'Waldwasserläufer',
  'g.', 'Green Woodpecker', 'Grünspecht',
  'gr', 'Greenfinch', 'Grünfink',
  'gk', 'Greenshank', 'Grünschenkel',
  'h.', 'Grey Heron', 'Graureiher',
  'p.', 'Grey Partridge', 'Rebhuhn',
  'gv', 'Grey Plover', 'Kiebitzregenpfeifer',
  'gl', 'Grey Wagtail', 'Gebirgsstelze',
  'gj', 'Greylag Goose', 'Graugans',
  'fw', 'Helmeted Guineafowl', 'Helmperlhuhn',
  'hf', 'Hawfinch', 'Kernbeißer',
  'hh', 'Hen Harrier', 'Kornweihe',
  'hg', 'Herring Gull', 'Silbermöwe',
  'hy', 'Hobby', 'Baumfalke',
  'hz', 'Honey Buzzard', 'Wespenbussard',
  'hc', 'Hooded Crow', 'Nebelkrähe',
  'hp', 'Hoopoe', 'Wiedehopf',
  'hm', 'House Martin', 'Mehlschwalbe',
  'hs', 'House Sparrow', 'Haussperling',
  'jd', 'Jackdaw', 'Dohle',
  'j.', 'Jay', 'Jay',
  'k.', 'Kestrel', 'Turmfalke',
  'kf', 'Kingfisher', 'Eisvogel',
  'ki', 'Kittiwake', 'Dreizehenmöwe',
  'kn', 'Knot', 'Knut',
  'lm', 'Lady Amhersts Pheasant', 'Diamantfasan',
  'la', 'Lapland Bunting', 'Spornammer',
  'l.', 'Lapwing', 'Kiebitz',
  'tl', 'Leachs Petrel', 'Wellenläufer',
  'lb', 'Lesser Black-backed Gull', 'Heringsmöwe',
  'ls', 'Lesser Spotted Woodpecker', 'Kleinspecht',
  'lw', 'Lesser Whitethroat', 'Klappergrasmücke',
  'li', 'Linnet', 'Bluthänfling',
  'et', 'Little Egret', 'Seidenreiher',
  'lg', 'Little Grebe', 'Zwergtaucher',
  'lu', 'Little Gull', 'Zwergmöwe',
  'lo', 'Little Owl', 'Steinkauz',
  'lp', 'Little Ringed Plover', 'Flussregenpfeifer',
  'af', 'Little Tern', 'Zwergseeschwalbe',
  'le', 'Long-eared Owl', 'Waldohreule',
  'lt', 'Long-tailed Tit', 'Schwanzmeise',
  'mg', 'Magpie', 'Elster',
  'ma', 'Mallard', 'Stockente',
  'mn', 'Mandarin Duck ', 'Mandarinente',
  'mx', 'Manx Shearwater', 'Atlantiksturmtaucher',
  'mr', 'Marsh Harrier', 'Rohrweihe',
  'mt', 'Marsh Tit', 'Sumpfmeise',
  'mw', 'Marsh Warbler', 'Sumpfrohrsänger',
  'mp', 'Meadow Pipit', 'Wiesenpieper',
  'mu', 'Mediterranean Gull', 'Schwarzkopfmöwe',
  'ml', 'Merlin', 'Merlin',
  'm.', 'Mistle Thrush', 'Misteldrossel',
  'mo', 'Montagus Harrier', 'Wiesenweihe',
  'mh', 'Moorhen', 'Teichhuhn',
  'ms', 'Mute Swan', 'Höckerschwan',
  'n.', 'Nightingale', 'Nachtigall',
  'nj', 'Nightjar', 'Nachtschwalbe',
  'nh', 'Nuthatch', 'Kleiber',
  'op', 'Osprey', 'Fischadler',
  'oc', 'Oystercatcher', 'Austernfischer',
  'px', 'Peafowl/Peacock', 'Pfau',
  'pe', 'Peregrine', 'Wanderfalke',
  'ph', 'Pheasant', 'Jagdfasan',
  'pf', 'Pied Flycatcher', 'Trauerschnäpper',
  'pw', 'Pied Wagtail', 'Bachstelze',
  'pg', 'Pink-footed Goose', 'Kurzschnabelgans',
  'pt', 'Pintail', 'Spießente',
  'po', 'Pochard', 'Tafelente',
  'pm', 'Ptarmigan', 'Schneehuhn',
  'pu', 'Puffin', 'Papageientaucher',
  'ps', 'Purple Sandpiper', 'Meerstrandläufer',
  'q.', 'Quail', 'Wachtel',
  'rn', 'Raven', 'Kolkrabe',
  'ra', 'Razorbill', 'Tordalk',
  'rg', 'Red Grouse', 'Moorschneehuhn',
  'kt', 'Red Kite', 'Rotmilan',
  'ed', 'Red-backed Shrike', 'Neuntöter',
  'rm', 'Red-breasted Merganser', 'Mittelsäger',
  'rq', 'Red-crested Pochard', 'Kolbenente',
  'fv', 'Red-footed Falcon', 'Rotfußfalke',
  'rl', 'Red-legged Partridge', 'Rothuhn',
  'nk', 'Red-necked Phalarope', 'Odinshühnchen',
  'lr', 'Redpoll', 'Alpenbirkenzeisig',
  'rk', 'Redshank', 'Rotschenkel',
  'rt', 'Redstart', 'Gartenrotschwanz',
  'rh', 'Red-throated Diver', 'Sterntaucher',
  're', 'Redwing', 'Rotdrossel',
  'rb', 'Reed Bunting', 'Rohrammer',
  'rw', 'Reed Warbler', 'Teichrohrsänger',
  'rz', 'Ring Ouzel', 'Ringdrossel',
  'rp', 'Ringed Plover', 'Sandregenpfeifer',
  'ri', 'Ring-necked Parakeet', 'Halsbandsittich',
  'r.', 'Robin', 'Rotkehlchen',
  'dv', 'Rock Dove', 'Felsentaube',
  'rc', 'Rock Pipit', 'Strandpieper',
  'ro', 'Rook', 'Saatkrähe',
  'rs', 'Roseate Tern', 'Rosenseeschwalbe',
  'ry', 'Ruddy Duck', 'Schwarzkopf-Ruderente',
  'ru', 'Ruff', 'Kampfläufer',
  'sm', 'Sand Martin', 'Uferschwalbe',
  'ss', 'Sanderling', 'Sanderling',
  'te', 'Sandwich Tern', 'Brandseeschwalbe',
  'vi', 'Savis Warbler', 'Rohrschwirl',
  'sq', 'Scarlet Rosefinch', 'Karmingimpel',
  'sp', 'Scaup', 'Bergente',
  'cy', 'Scottish Crossbill', 'Schottischer Fichtenkreuzschnabel',
  'sw', 'Sedge Warbler', 'Schilfrohrsänger',
  'ns', 'Serin', 'Girlitz',
  'sa', 'Shag', 'Krähenscharbe',
  'su', 'Shelduck', 'Brandgans',
  'sx', 'Shorelark', 'Ohrenlerche',
  'se', 'Short-eared Owl', 'Sumpfohreule',
  'sv', 'Shoveler', 'Löffelente',
  'sk', 'Siskin', 'Erlenzeisig',
  's.', 'Skylark', 'Feldlerche',
  'sz', 'Slavonian Grebe', 'Ohrentaucher',
  'sn', 'Snipe', 'Bekassine',
  'sb', 'Snow Bunting', 'Schneeammer',
  'st', 'Song Thrush', 'Singdrossel',
  'sh', 'Sparrowhawk', 'Sperber',
  'ak', 'Spotted Crake', 'Tüpfelsumpfhuhn',
  'sf', 'Spotted Flycatcher', 'Grauschnäpper',
  'dr', 'Spotted Redshank', 'Dunkelwasserläufer',
  'sg', 'Starling', 'Star',
  'sd', 'Stock Dove', 'Hohltaube',
  'sc', 'Stonechat', 'Schwarzkehlchen',
  'tn', 'Stone-curlew', 'Triel',
  'tm', 'Storm Petrel', 'Sturmschwalbe',
  'sl', 'Swallow', 'Rauchschwalbe',
  'si', 'Swift', 'Mauersegler',
  'to', 'Tawny Owl', 'Waldkauz',
  't.', 'Teal', 'Krickente',
  'tk', 'Temmincks Stint', 'Temminckstrandläufer',
  'tp', 'Tree Pipit', 'Baumpieper',
  'ts', 'Tree Sparrow', 'Feldsperling',
  'tc', 'Eurasian Treecreeper', 'Waldbaumläufer',
  'tu', 'Tufted Duck', 'Reiherente',
  'tt', 'Turnstone', 'Steinwälzer',
  'td', 'Turtle Dove', 'Turteltaube',
  'tw', 'Twite', 'Berghänfling',
  'wa', 'Water Rail', 'Wasserralle',
  'w.', 'Wheatear', 'Steinschmätzer',
  'wm', 'Whimbrel', 'Regenbrachvogel',
  'wc', 'Whinchat', 'Braunkehlchen',
  'wg', 'White-fronted Goose', 'Blässgans',
  'wh', 'Whitethroat', 'Dorngrasmücke',
  'ws', 'Whooper Swan', 'Singschwan',
  'wn', 'Wigeon', 'Pfeifente',
  'wt', 'Willow Tit', 'Weidenmeise',
  'ww', 'Willow Warbler', 'Fitis',
  'od', 'Wood Sandpiper', 'Bruchwasserläufer',
  'wo', 'Wood Warbler', 'Waldlaubsänger',
  'wk', 'Woodcock', 'Waldschnepfe',
  'wl', 'Woodlark', 'Heidelerche',
  'wp', 'Woodpigeon', 'Ringeltaube',
  'wr', 'Wren', 'Zaunkönig',
  'wy', 'Wryneck', 'Wendehals',
  'yw', 'Yellow Wagtail', 'Schafstelze',
  'y.', 'Yellowhammer', 'Goldammer']


def filename_to_annotations(filename):
    filename = filename[:-4]
    annotations = filename.split("_")
    annotations = [an for an in annotations if an != ""]
    annotations[2] = annotations[2][:-3]
    annotations[3] = annotations[3][:-1]
    annotations[2] = annotations[2].split("-")
    annotations[3] = annotations[3].split("-")
    # flatten the list:
    annotations = [item for sublist in annotations for item in (sublist if isinstance(sublist, list) else [sublist])]
    annotations = [filename] + annotations

    species_code = annotations[-1]

    species_dict = {species_mapping[i]: (species_mapping[i+1], species_mapping[i+2]) for i in range(0, len(species_mapping), 3)}
    
    if species_code in species_dict:
        values = species_dict[species_code]
        annotations.append('/'.join(values))

    #converted_datetime = julian_to_datetime(float(annotations[1]))
    #annotations.insert(3, converted_datetime)

    return annotations

def append_rows_tocsv(csv_filepath):
    title = ["filename", "juliandate", "loc", "low_freq", "high_freq", "start", "end", "species", "species_eng", "species_ger"]
    audio_filepaths = [file for file in os.listdir("data/dataset/Audiodateien") if file.endswith('.wav')]
    with open(csv_filepath, "w", newline="") as annotations_file:
        wr = csv.writer(annotations_file)
        wr.writerow(title)
        for file in audio_filepaths:
            annotation = filename_to_annotations(file)
            wr.writerow(annotation)

umlaut_mapping = {
    'ä': 'ae',
    'ö': 'oe',
    'ü': 'ue',
    'Ä': 'Ae',
    'Ö': 'Oe',
    'Ü': 'Ue',
    'ß': 'ss'
}

def replace_umlauts(text):
    if isinstance(text, str):
        for umlaut, replacement in umlaut_mapping.items():
            text = text.replace(umlaut, replacement)
    return text
    

if __name__ == "__main__":
    
    append_rows_tocsv("data/annotations.csv")

    # Try reading the CSV file with a different encoding
    try:
        df = pd.read_csv("data/annotations.csv", encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv("data/annotations.csv", encoding='latin1')

    df = df.map(replace_umlauts)
    
    # Write the DataFrame back to a CSV file without changes
    df.to_csv("data/annotations.csv", index=False, encoding='utf-8')

