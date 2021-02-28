.. highlight:: shell

============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every
little bit helps, and credit will always be given.

You can contribute in many ways:

Types of Contributions
----------------------

Report Bugs
~~~~~~~~~~~

Report bugs at https://github.com/dPys/pynets/issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

Fix Bugs
~~~~~~~~

Look through the GitHub issues for bugs. Anything tagged with "bug"
and "help wanted" is open to whoever wants to implement it.

Implement Features
~~~~~~~~~~~~~~~~~~

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

Write Documentation
~~~~~~~~~~~~~~~~~~~

PyNets could always use more documentation, whether as part of the
official PyNets docs, in docstrings, or even on the web in blog posts,
articles, and such.

Submit Feedback
~~~~~~~~~~~~~~~

The best way to send feedback is to file an issue at https://github.com/dPys/pynets/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

Submit Pull-Requests using informative commit prefixes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    [ENH]: enhancements or new features.
    [FIX]: bug fixes.
    [TST]: new or updated tests.
    [DOC]: new or updated documentation.
    [STY]: style changes.
    [REF]: refactoring existing code.
    [CI]: updates to continuous integration infrastructure.
    [MAINT]: general maintenance.
    [WIP]: works-in-progress.

Get Started!
------------

Ready to contribute? Here's how to set up `pynets` for local development.

1. Fork the `pynets` repo on GitHub.
2. Clone your fork locally::

    $ git clone git@github.com:your_name_here/pynets.git

3. Configure your git remotes::

    $ git remote add origin https://github.com/your_name_here/pynets.git
    $ git remote add upstream https://github.com/dPys/pynets.git
    $ git remote -v

4. Install your local copy::

    $ cd pynets/
    $ python setup.py develop

5. Create a feature branch for local development::

    $ git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally.

6. When you're done making changes, ensure you are still in-sync with any
   upstream changes that may have happened in the meantime::

    $ git fetch upstream
    $ git merge upstream/development

7. It's also a good idea to check that your changes pass tests::

    $ pytest -vvv

7. Commit your changes and push your branch to GitHub::

    $ git add .
    $ git commit -m "[PREFIX] Your detailed description of your changes."
    $ git push -u origin name-of-your-bugfix-or-feature

8. Submit a pull request through the GitHub website.

Pull Request Guidelines
-----------------------

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring, and add the
   feature to the list in README.rst.
3. The pull request should work for Python 3.6+, and for PyPy. Check
   https://app.circleci.com/pipelines/github/dPys/PyNets
   and make sure that the tests pass for all supported Python versions.


