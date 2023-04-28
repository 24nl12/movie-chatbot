"""
Please answer the following ethics and reflection questions. 

We are expecting at least three complete sentences for each question for full credit. 

Each question is worth 2.5 points. 
"""

######################################################################################
"""
QUESTION 1 - Anthropomorphizing

Is there potential for users of your chatbot possibly anthropomorphize 
(attribute human characteristics to an object) it? 
What are some possible ramifications of anthropomorphizing chatbot systems? 
Can you think of any ways that chatbot designers could ensure that users can easily 
distinguish the chatbot responses from those of a human?
"""

Q1_your_answer = """

Yes, there is because it is having a discussion with the user in a somewhat natural manner. This could have ramifications on social media
where posts, messages, and pages could be populated by chatbots and could go completely unnoticed when interacting with humans online. This could
lead to greater distrust on social media or certain organizations using chatbots to push a certain harmful viewpoint by acting like human actors. Designers
of chatbots could counter act this by giving their chatbot a signature that allows people to know they are not humans. In addition, designers could
design their own algorithm to detect if a piece of text is generated by their chatbot. This could be difficult to enforce and designers would have to be willing
to build one for their chatbots.

"""

######################################################################################

"""
QUESTION 2 - Privacy Leaks

One of the potential harms for chatbots is collecting and then subsequently leaking 
(advertly or inadvertently) private information. Does your chatbot have risk of doing so? 
Can you think of ways that designers of the chatbot can help to mitigate this risk? 
"""

Q2_your_answer = """

Yes it definitely does. Even though the chatbot is only collecting information on movies, there is still information on people's preferences
that could be used against their consent. We are storing what they like and dislike which could lead to corporations wanting data on this type of list. 
In order to mitigate this risk, investment in cyber security infrastructure is obviously very important to prevent any hackers from accessing people's sensitive data.
Additionally, designers of chatbots could anonymously gather information on users and make sure that any sensitive information is not held for long periods of time to make 
it less suceptible of a harmful data breach.

"""

######################################################################################

"""
QUESTION 3 - Effects on Labor

Advances in dialogue systems, and the language technologies based on them, could lead to the automation of 
tasks that are currently done by paid human workers, such as responding to customer-service queries, 
translating documents or writing computer code. These could displace workers and lead to widespread unemployment. 
What do you think different stakeholders -- elected government officials, employees at technology companies, 
citizens -- should do in anticipation of these risks and/or in response to these real-world harms? 
"""

Q3_your_answer = """

There is a lot of possibilities for government to regulate the extent to which we are able to rely on chatbots and language technologies. Although
they are helpful and can serve many purposes, destroying entire sectors could lead to much social unrest and political turmoil. Therefore, it is important
for governments to intervene to create a gradual transition if this does occur, while also assisting those displaced by offering training and other measures that could
shift their skills to other valuable sectors. Employees at technology companies should be made aware of the effects of their projects and the potential economic and political
consequences that these new dialogue systems can have. NLP's priority should be in advancing society not dominating or controlling it. Thus, it falls on tech companies and those
who work at these corporations to take responsibility in their work, prioritizing safety over progress.

"""

"""
QUESTION 4 - Refelection 

You just built a frame-based dialogue system using a combination of rule-based and machine learning 
approaches. Congratulations! What are the advantages and disadvantages of this paradigm 
compared to an end-to-end deep learning approach, e.g. ChatGPT? 
"""

Q4_your_answer = """

The advantages of this is that the modules are very easy to separately understand. Each function has a clear goal and with each step of building
this frame-based dialogue you are stacking one block on top of the other. This is very helpful in debugging and adding new functions. You can often tweak a 
certain function or piece without causing detrimental issues to the entire system. The disadvantage is it is not very generalizable. The chatbot is only able to handle
a hand full of scenarios and gives predictable, deterministic responses. These all contrast an end-to-end deep learning approach which can often be very complex and a "black box." Deep
learning can be very difficult to debug and implement because it is stochastic. However, it is very generalizable and is able to handle many user inputs
and respond with a possibility of outputs. 

"""