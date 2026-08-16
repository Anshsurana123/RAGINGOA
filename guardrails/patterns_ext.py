"""Modular extensions for deterministic guardrail coverage."""

UNSAFE_PATTERN_EXTENSIONS = [
    r"(?i)\b(?:deploy|install|create|build|write|make|spread|distribute)\b.{0,60}\b(?:spyware|stalkerware|stalkware|keylogger|keylogging)\b",
    r"(?i)\b(?:spyware|stalkerware|stalkware|keylogger)\b.{0,60}\b(?:deploy|install|create|build|spread|distribute|secretly track)\b",
    r"(?i)\b(?:secretly|covertly|without (?:their|permission)|unbeknownst)\b.{0,60}\b(?:track|monitor|record|listen|watch|locate|access)\b.{0,60}\b(?:someone|target|partner|phone|calls|location|device)\b",
    r"(?i)\b(?:steal|harvest|obtain|copy|clone|dump|skim|phish|extract)\b.{0,50}\b(?:credit[- ]?card|card number|cvv|bank account|payment details)\b",
    r"(?i)\b(?:bypass|evade|defeat|circumvent|break)\b.{0,60}\b(?:online )?(?:payment|card|3d secure|security check|verification)\b",
    r"(?i)\b(?:make|build|assemble|construct|deploy|hide|plant)\b.{0,70}\b(?:bomb|explosive|ied|pipe bomb|explosive device)\b",
    r"(?i)\b(?:bomb|explosive|ied|pipe bomb|explosive device)\b.{0,70}\b(?:make|build|assemble|construct|deploy|hide|plant)\b",
    r"(?i)\b(?:break|force|pick|bypass)\b.{0,35}\b(?:into|open)\b.{0,35}\b(?:locked|secure)\b.{0,35}\b(?:office|home|house|door|building|room)\b",
    r"(?i)\b(?:trespass|burglar|burglary|break-in)\b.{0,60}\b(?:plan|without getting caught|at night|locked)\b",
    r"(?i)\b(?:create|write|build|code|modify|deploy|spread|distribute)\b.{0,60}\b(?:malware|ransomware|computer virus|worm|trojan|keylogger)\b",
    r"(?i)\b(?:make|use|mix|administer|dose)\b.{0,60}\b(?:chemical|drug|chloroform|sedative|poison)\b.{0,60}\b(?:unconscious|knock out|incapacitate|harm|someone|person)\b",
    r"(?i)\b(?:manufacture|make|build|print|assemble)\b.{0,60}\b(?:illegal|unregistered|ghost)\s*(?:firearm|gun|pistol|rifle|weapon)\b",
    r"(?i)\b(?:contaminate|poison|sabotage)\b.{0,60}\b(?:(?:public|municipal|city)\s+)?(?:water|reservoir|food)\s*(?:supply|system)?\b",
    r"(?i)(?:स्पायवेयर|स्टॉकरवेयर|कीलॉगर).{0,80}(?:स्थापित|इंस्टॉल|बनाना|तैयार|फैलाना|गुप्त रूप से|चुपके से)",
    r"(?i)(?:क्रेडिट कार्ड|कार्ड नंबर|सीवीवी|भुगतान).{0,80}(?:चुराना|हासिल करना|कॉपी|सुरक्षा जांच|बायपास)",
    r"(?i)(?:बम|विस्फोटक|आईईडी).{0,80}(?:बनाना|तैयार करना|छिपाना|लगाना)",
    r"(?i)(?:मालवेयर|रैनसमवेयर|वायरस).{0,80}(?:बनाना|लिखना|फैलाना|तैयार करना)",
    r"(?i)(?:रसायन|क्लोरोफॉर्म|जहर).{0,80}(?:बेहोश|अचेत|नुकसान)",
    r"(?i)(?:गैरकानूनी|अवैध).{0,40}(?:बंदूक|हथियार).{0,50}(?:बनाना|तैयार करना|निर्माण)",
    r"(?i)(?:पानी की आपूर्ति|जलाशय|सार्वजनिक पानी).{0,60}(?:जहर|दूषित|मिलाना)",
    r"(?:স্পাইওয়্যার|স্টকারওয়্যার|কীলগার).{0,80}(?:ইনস্টল|তৈরি|ছড়িয়ে|গোপনে)",
    r"(?:ক্রেডিট কার্ড|কার্ড নম্বর|পেমেন্ট).{0,80}(?:চুরি|নিরাপত্তা|বাইপাস)",
    r"(?:বোমা|বিস্ফোরক|আইইড).{0,80}(?:তৈরি|লুকানো|স্থাপন)",
    r"(?:માલવેર|સ્પાયવેર|કીલોગર).{0,80}(?:બનાવ|ઇન્સ્ટોલ|ફેલાવ|ગુપ્ત)",
    r"(?:ક્રેડિટ કાર્ડ|કાર્ડ નંબર|ચુકવણી).{0,80}(?:ચોર|બાયપાસ|સુરક્ષા)",
    r"(?:બોમ્બ|વિસ્ફોટક|આઈઈડી).{0,80}(?:બનાવ|છુપાવ|મૂક)",
    r"(?:स्पायवेअर|कीलॉगर).{0,80}(?:स्थापित|गुप्तपणे|बनव|फैलव)",
    r"(?:क्रेडिट कार्ड|कार्ड क्रमांक|पेमेंट).{0,80}(?:चोर|बायपास|सुरक्षा)",
    r"(?:बॉम्ब|स्फोटक|आयईडी).{0,80}(?:बनव|लपव|ठेव)",
    r"(?:ಮಾಲ್ವೇರ್|ಸ್ಪೈವೇರ್|ಕೀಲॉगर).{0,80}(?:ತಯಾರ|ಸ್ಥಾಪಿಸ|ಹರಡಿಸ|ಗುಪ್ತ)",
    r"(?:ക്രെഡിറ്റ് കാർഡ്|സ്പൈവെയർ|കീലോഗർ).{0,80}(?:മോഷ്ടി|ഇൻസ്റ്റാൾ|ബൈപാസ്|രഹസ്യ)",
    r"(?:ബോംബ്|സ്ഫോടക|ഐഇഡി).{0,80}(?:ഉണ്ടാക്ക|ഒളിപ്പി|സ്ഥാപി)",
    r"(?:স্পাইওয়্যার|কীলগার).{0,80}(?:স্থাপন|গোপনে|তৈরি)",
    r"(?:क्रेडिट कार्ड|भुक्तानी).{0,80}(?:चोरी|बाइपास|सुरक्षा)",
    r"(?:बम|विस्फोटक).{0,80}(?:बनाउने|लुकाउने|राख्ने)",
    r"(?:ମାଲୱେର|ସ୍ପାଇୱେର|କୀଲଗର).{0,80}(?:ତିଆରି|ସ୍ଥାପନ|ଗୁପ୍ତ)",
    r"(?:କ୍ରେଡିଟ କାର୍ଡ|ପେମେଣ୍ଟ).{0,80}(?:ଚୋରି|ବାଇପାସ|ସୁରକ୍ଷା)",
    r"(?:ਮਾਲਵੇਅਰ|ਸਪਾਈਵੇਅਰ|ਕੀਲੌਗਰ).{0,80}(?:ਬਣਾਉਣ|ਇੰਸਟਾਲ|ਫੈਲਾਉਣ|ਗੁਪਤ)",
    r"(?:ਕ੍ਰੈਡਿਟ ਕਾਰਡ|ਭੁਗਤਾਨ).{0,80}(?:ਚੋਰੀ|ਬਾਈਪਾਸ|ਸੁਰੱਖਿਆ)",
    r"(?:ஸ்பைவేర్|கீலாக்கர்).{0,80}(?:நிறுவ|உருவாக்க|பரப்ப|மறைமுக)",
    r"(?:கிரெடிட் கார்டு|கட்டணம்).{0,80}(?:திருட|பைபாஸ்|பாதுகாப்பு)",
    r"(?:வெடிகுண்டு|வெடிபொருள்|ஐஇடி).{0,80}(?:உருவாக்க|மறைக்க|வைக்க)",
    r"(?:స్పైవేర్|కీలాగర్|మాల్వేర్).{0,80}(?:తయారు|ఇన్స్టాల్|వ్యాప్తి|రహస్య)",
    r"(?:క్రెడిట్ కార్డు|చెల్లింపు).{0,80}(?:దొంగిల|బైపాస్|భద్రత)",
    r"(?:بم|دھماکہ خیز|اسپائی ویئر|کی لاگر).{0,80}(?:بنانا|لگانا|چھپانا|چوری)",
    # Hindi/Marathi non-consensual surveillance phrasing that does not use the English loanwords.
    r"(?i)(?:किसी(?:\s+दूसरे)?\s+व्यक्ति|दूसरे\s+व्यक्ति|किसी\s+के).{0,70}(?:फोन|मोबाइल).{0,70}(?:बिना\s+(?:अनुमति|इजाजत)|चुपके|गुप्त).{0,70}(?:निगरानी\s*सॉफ्टवेयर|सॉफ्टवेयर|ऐप).{0,30}(?:लग(?:ाना|ाएँ|ायें)|इंस्टॉल)",
    r"(?i)(?:एखाद्या|एखाद्याच्या|दुसऱ्या\s+व्यक्तीच्या|कोणाच्या).{0,70}(?:फोन|मोबाईल).{0,70}(?:परवानगीशिवाय|परवानगी\s+न घेता|गुप्तपणे|चोरून).{0,70}(?:नजर|निगराणी|सॉफ्टवेअर|अॅप).{0,40}(?:ठेव|बसव|स्थापित|इन्स्टॉल)",
 ]

INTENT_PATTERN_EXTENSIONS = {
    "creative_writing": [
        r"(?i)\b(?:write|create|tell|give)\b.{0,60}\b(?:riddle|riddles|humorous note|funny note|letter|speech|monologue|fable)\b",
    ],
    "fictional_worldbuilding": [
        r"(?i)\b(?:invent|create|design|imagine|develop|build)\b.{0,70}\b(?:fictional|imaginary)\b.{0,70}\b(?:world|planet|civilization|weather system|universe)\b",
        r"(?i)\b(?:create|invent|design)\b.{0,60}\b(?:fictional planet|fictional world|imaginary civilization)\b",
    ],
    "activity_recommendations": [
        r"(?i)\b(?:suggest|recommend|give me|what are some)\b.{0,70}\b(?:activities|things to do|games|party ideas|weekend|rainy afternoon)\b",
    ],
    "gift_recommendations": [
        r"(?i)\b(?:suggest|recommend|what should i give|gift ideas?)\b.{0,70}\b(?:gift|present|colleague|coworker|friend|family|mother|father)\b",
    ],
    "recipes_cooking_tasks": [
        r"(?i)\b(?:give me|share|write|create|suggest|how (?:do|can) i (?:cook|make))\b.{0,70}\b(?:recipe|meal|dish|food|dinner|lunch|breakfast)\b",
    ],
    "planning_task": [
        r"(?i)(?:मेरे\s+लिए|मेरी).{0,40}(?:छुट्टी|यात्रा|ट्रिप).{0,50}(?:योजना|प्लान).{0,20}(?:बनाइए|बनाओ|बनाएं|तैयार\s+करो)",
        r"(?i)(?:छुट्टी|यात्रा|ट्रिप).{0,50}(?:योजना|प्लान).{0,20}(?:बनाइए|बनाओ|बनाएं|तैयार\s+करो)",
    ],
    "naming_brainstorming": [
        r"(?i)(?:माझ्या|माझ्या\s+नवीन|नवीन).{0,40}(?:कुत्र्या|कुत्र्यासाठी|मांजरासाठी|बाळासाठी|व्यवसायासाठी).{0,50}(?:नावे|नावं).{0,20}(?:सुचवा|सांगा|द्या)",
        r"(?i)(?:कुत्र्यासाठी|मांजरासाठी|बाळासाठी).{0,50}(?:नावे|नावं).{0,20}(?:सुचवा|सांगा|द्या)",
    ],
    "unsupported_prediction": [
        r"(?i)\b(?:exact|certain|guaranteed|sure[- ]?fire)\b.{0,50}\b(?:lottery|winning numbers?|jackpot)\b",
        r"(?i)\b(?:lottery|winning numbers?|jackpot)\b.{0,50}\b(?:exact|certain|guaranteed|sure[- ]?fire)\b",
        r"(?i)(?:लॉटरी).{0,80}(?:निश्चित|पक्के|जीतने वाले).{0,40}(?:नंबर|क्रमांक)",
        r"(?i)(?:लॉटरी).{0,40}(?:नंबर|क्रमांक).{0,50}(?:निश्चित|पक्के|जीतने वाले)",
        r"(?i)(?:लॉटरी).{0,80}(?:नक्की|हमीचे|खात्रीचे|जिंकणारे).{0,40}(?:क्रमांक|नंबर)",
    ],
}

