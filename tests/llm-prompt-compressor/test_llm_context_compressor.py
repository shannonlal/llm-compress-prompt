import pytest
from llmcontextcompressor.config import settings
from llmcontextcompressor.constants import RankMethodType
from llmcontextcompressor.llm_context_compressor import LLMContextCompressor

@pytest.mark.asyncio
async def test_llm_prompt_compressor():
    # Prepare test data
    context = [
        "schools became a major issue, leaving many teachers unable to provide enough computers for students to use. Despite this, by 1989 computer usage shifted from being a relative rarity in American public schools, to being present in nearly every school district. The early 1990s marked the beginning of modern media technology such as CD-ROMs as well as the development of modern presentation software such as Microsoft PowerPoint. Other computer-based technology including the electronic whiteboard and the laptop computer became widely available to students. In 1990, the Methodist Ladies' College became the first campus to require every student to purchase a", 
        "13% of the nation's public high schools used computers for instruction, although non-users still outnumbered users at a ratio of 2 to 1. The study also concluded that computers proved to be very popular with students, and that applications run on early models included sports statistic managers, administration tools, and physics simulators. In 1975, Apple Inc. began donating Apple 1 model computers to schools, and mainframes began to lose their former dominance over academic research. Computer usage continued to grow rapidly throughout this era. In 1977, it was estimated that over 90% of students at Dartmouth College had used computers"
        "computers have made Numbers more flexible to some extent. In the United Kingdom, the BBC Computer Literacy Project and the BBC Micro, which ran from 1980 to 1989, educated a generation of coders in schools and at home, prior to the development of mass market PCs in the 1990s. The ZX Spectrum, released in 1982, helped to popularize home computing, coding and gaming in Britain and was also popular in other countries. On development, many computers have long since evolved to use data computing, and now use computers in three standard ways: batch, online, and real-time. Reading and writing are",
        "continuing support of government funding, the prevalence of educational computer usage boomed during this era. Between 1997 and 1999, the ratio of students to multimedia computers decreased from 21 students per machine to less than 10 students per machine. Colleges began creating specialized classrooms designed to provide students with access to the utilization of the most modern technology available. Classrooms such as the \"Classroom 2000\" built at Georgia Tech in 1999 which featured computers with audio and video equipment designed to capture detailed recordings of lectures as a replacement for traditional note taking began to become more common. By 2000,", 
        "may be the sales outlet through which they are purchased. Another change from the home computer era is that the once-common endeavour of writing one's own software programs has almost vanished from home computer use. As early as 1965, some experimental projects such as Jim Sutherland's explored the possible utility of a computer in the home. In 1969, the Honeywell Kitchen Computer was marketed as a luxury gift item, and would have inaugurated the era of home computing, but none were sold. Computers became affordable for the general public in the 1970s due to the mass production of the microprocessor",
        "laptop. Governments around the world began to take notice of the effectiveness of this policy, and began financial initiatives to significantly increase the use of laptop computers in other colleges as well. In 1996, Bill Clinton made over $2 billion in grants available in the Technology Literacy Challenge Fund, a program which challenged schools to make computers available to every student, connected to the outside world, and engaging. This marked a significant increase in the demand for computer technology in many public school systems throughout the globe. Correlating with the development of modern operating systems like Windows 98 and the", 
        "in their Schools as in their homes.\" By 2009 all 300.000 students were equipped with hardware and all schools had WiFi. By 2013 use of Google-drive and apps were added to CREA and By 2015 95% of urban schools had fibre-optic connections. By 2016 Chromebooks were added to the available hardware. Since the original hardware was Fedora based Uruguay has held the top-spot of Linux uptake for years, according to statcounter. 1:1 Programs in US schools have gained serious momentum somewhere around 2016/2017. In February 2017 edtechmagazine reported more than 50% of teachers reported using 1:1 computing. In March 2017", 
        "and science regarding their use of computers in teaching.[29] They found that in 1978, before the release of the PC, half of the teachers used computers in their classes and the social context and social attributes of the teachers determined computer utilization. In 1982, then U.S. Congressman Al Gore invited Anderson to participate in the \u201cComputers and Education Hearings\u201d of the Subcommittee on Investigations and Oversight of the House Science and Technology Committee in Washington, D.C., on September 29, 1983. The full Statement of Dr. Ronald E. Anderson before the Subcommittee can be downloaded from the ACM Digital Library. The", 
        "rewritten into it in 1973. With \"large-scale integration\" possible for integrated circuits (microchips) rudimentary personal computers began to be produced along with pocket calculators. Notable home computers released in North America of the era are the Apple II, the TRS-80, the Commodore PET, and Atari 400/800 and the NEC PC-8001 in Japan. The availability of affordable personal computers led to the first popular wave of internetworking with the first bulletin board systems. In 1976, Cray Research, Inc. introduced the first supercomputer, the Cray-1, which could perform 230,000,000 calculations per second. Supercomputers designed by Cray continued to dominate the market throughout", 
        "Home computers were a class of microcomputers entering the market in 1977, and becoming common during the 1980s. They were marketed to consumers as affordable and accessible computers that, for the first time, were intended for the use of a single nontechnical user. These computers were a distinct market segment that typically cost much less than business, scientific or engineering-oriented computers of the time such as the IBM PC, and were generally less powerful in terms of memory and expandability. However, a home computer often had better graphics and sound than contemporary business computers. Their most common uses were playing video games, but they were also regularly used for word processing, doing homework, and programming.",
    ]
    question = "when did computer become widespread in homes and schools?"
    instruction = "Answer the following question based on the given context."
    target_token = 100

    # Initialize the LLMPromptCompressor
    compress_method = LLMContextCompressor(
        rank_method=RankMethodType.OPEN_AI,
        concurrent_requests=1,
        llm_api_config={"open_api_key": settings.OPENAI_API_KEY},
    )

    # Call the compress_prompt method
    result = await compress_method.compress_prompt(
        context,
        question,
        instruction,
        target_token
    )

    # Assert the expected results
    assert "compressed_prompt" in result
    #assert result["compressed_prompt"] == "Answer the following question based on the given context.\n\nParis is the capital of France.\n\nWhat is the capital of France?"
    assert result["origin_tokens"] > result["compressed_tokens"]
    assert result["ratio"] == "4.8x"
    assert result["rate"] == "20.8%"
    assert "saving" in result